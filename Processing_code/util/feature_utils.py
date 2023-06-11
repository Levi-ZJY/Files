from Bio.PDB import PDBParser
import pandas as pd
import numpy as np
import os
import rdkit.Chem as Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from tqdm import tqdm
import glob
import torch
import torch.nn.functional as F
from io import StringIO
import sys
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.PDBIO import Select
import scipy
import scipy.spatial
import requests
from rdkit.Geometry import Point3D

#from rdkit.Chem import WrapLogs

from torchdrug import data as td     # conda install torchdrug -c milagraph -c conda-forge -c pytorch -c pyg if fail to import

def read_mol(sdf_fileName, mol2_fileName, verbose=False):
    Chem.WrapLogs()
    stderr = sys.stderr
    sio = sys.stderr = StringIO()
    mol = Chem.MolFromMolFile(sdf_fileName, sanitize=False)   #先尝试读取.sdf文件
    problem = False
    try:
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
        sm = Chem.MolToSmiles(mol)
    except Exception as e:
        sm = str(e)
        problem = True
    if problem:                #如果.sdf文件无法读取，尝试读取.mol2文件
        mol = Chem.MolFromMol2File(mol2_fileName, sanitize=False)
        problem = False
        try:
            Chem.SanitizeMol(mol)
            mol = Chem.RemoveHs(mol)
            sm = Chem.MolToSmiles(mol)
            problem = False
        except Exception as e:     #如果两个文件都读不了，就返回错误提示
            sm = str(e)
            problem = True

    if verbose:
        print(sio.getvalue())
    sys.stderr = stderr
    return mol, problem


def write_renumbered_sdf(toFile, sdf_fileName, mol2_fileName):
    # read in mol
    mol, _ = read_mol(sdf_fileName, mol2_fileName)
    # reorder the mol atom number as in smiles.
    m_order = list(mol.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder'])  #用于获取分子对象（mol）的所有属性和值，并以字典的形式返回
    # TODO
    mol = Chem.RenumberAtoms(mol, m_order)      #用来对分子中的原子进行重新编号，以便与另一个分子或模板的原子顺序一致
    w = Chem.SDWriter(toFile)      #用于将分子对象（mol）写入SDF文件
    w.write(mol)
    w.close()

def get_canonical_smiles(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def generate_rdkit_conformation_v2(smiles, n_repeat=50):
    mol = Chem.MolFromSmiles(smiles)
    # mol = Chem.RemoveAllHs(mol)
    # mol = Chem.AddHs(mol)
    ps = AllChem.ETKDGv2()
    # rid = AllChem.EmbedMolecule(mol, ps)
    for repeat in range(n_repeat):
        rid = AllChem.EmbedMolecule(mol, ps)
        if rid == 0:
            break
    if rid == -1:
        print("rid", pdb, rid)
        ps.useRandomCoords = True
        rid = AllChem.EmbedMolecule(mol, ps)
        if rid == -1:
            mol.Compute2DCoords()
        else:
            AllChem.MMFFOptimizeMolecule(mol, confId=0)
    else:
        AllChem.MMFFOptimizeMolecule(mol, confId=0)
    # mol = Chem.RemoveAllHs(mol)
    return mol


def binarize(x):    # 将输入的张量x进行二值化处理。如果x中的元素大于0，则将其设置为1，否则将其设置为0。
    return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))
    # torch.ones_like函数用于创建一个与输入张量形状相同的张量，其中所有元素都设置为1


    # torch.eye函数用于创建一个二维矩阵，对角线上的元素为1，其余元素为0

def n_hops_adj(adj, n_hops):   # 此函数作用是求adj矩阵的n阶邻接矩阵
    adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device), binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]
    #adj - > n_hops connections adj

    for i in range(2, n_hops+1):
        adj_mats.append(binarize(adj_mats[i-1] @ adj_mats[1]))
    extend_mat = torch.zeros_like(adj)

    for i in range(1, n_hops+1):
        extend_mat += (adj_mats[i] - adj_mats[i-1]) * i

    return extend_mat

def get_LAS_distance_constraint_mask(mol):      # LAS距离约束掩码的原理是通过约束分子中原子之间的距离，来模拟分子中的化学反应、蛋白质折叠等过程。、
                                                # 具体来说，LAS距离约束掩码可以通过限制原子之间的距离，来模拟分子中的化学键、氢键等化学反应。
    # Get the adj
    adj = Chem.GetAdjacencyMatrix(mol)  # 用于构建分子的邻接矩阵，返回一个二维数组，数组中的每个元素表示分子中两个原子之间的键的数量
    adj = torch.from_numpy(adj)         # 将numpy转化为tensor格式
    extend_adj = n_hops_adj(adj,2)      # 求adj矩阵的2阶邻接矩阵
    # add ring
    ssr = Chem.GetSymmSSSR(mol)    # 获取分子中所有的环，以及每个环对应的原子组成
    for ring in ssr:
        # print(ring)
        for i in ring:
            for j in ring:
                if i==j:
                    continue
                else:
                    extend_adj[i][j]+=1    # 同一个环中的原子，每一对加一个有向边（除自己跟自己外）
    # turn to mask
    mol_mask = binarize(extend_adj)   # 二值化
    return mol_mask

def Seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_compound_pair_dis_distribution(coords, LAS_distance_constraint_mask=None):   # 函数首先计算原子对之间的距离，然后将距离分成若干个区间，统计每个区间内的原子对数量，最后返回原子对距离分布。
    
    pair_dis = scipy.spatial.distance.cdist(coords, coords)  # cdist用于计算两个集合中所有点之间的距离。
    bin_size=1       #这三行代码定义了距离分布的区间大小、最小值和最大值。
    bin_min=-0.5
    bin_max=15
    if LAS_distance_constraint_mask is not None:     # 这一段代码是对LAS距离约束掩码的处理，如果LAS_distance_constraint_mask为空，则将距离掩码为0（表示两个原子间无关系）的原子对的距离设置为最大距离bin_max；将对角线上的距离设置为0。
        pair_dis[LAS_distance_constraint_mask==0] = bin_max
        # diagonal is zero.
        for i in range(pair_dis.shape[0]):
            pair_dis[i, i] = 0
    pair_dis = torch.tensor(pair_dis, dtype=torch.float)  # 转为tensor张量
    pair_dis[pair_dis>bin_max] = bin_max                  # 将距离大于bin_max的原子对的距离设置为bin_max。
    pair_dis_bin_index = torch.div(pair_dis - bin_min, bin_size, rounding_mode='floor').long()    #  (pair_dis - bin_min)/bin_size，将距离的值离散分布到各个区间
    pair_dis_one_hot = torch.nn.functional.one_hot(pair_dis_bin_index, num_classes=16)    # 将离散距离分布转化为one_hot向量
    pair_dis_distribution = pair_dis_one_hot.float()      # 转为float
    return pair_dis_distribution


def extract_torchdrug_feature_from_mol(mol, has_LAS_mask=False):
    coords = mol.GetConformer().GetPositions()   # 该函数返回一个包含所有原子三维坐标的列表，列表中每个元素都是一个三元组，分别表示原子的x、y、z坐标
    if has_LAS_mask:
        LAS_distance_constraint_mask = get_LAS_distance_constraint_mask(mol)    # ！！！获取LAS距离约束掩码！！！
    else:
        LAS_distance_constraint_mask = None
    pair_dis_distribution = get_compound_pair_dis_distribution(coords, LAS_distance_constraint_mask=LAS_distance_constraint_mask)   #！！！获得原子对距离分布！！！

    molstd = td.Molecule.from_smiles(Chem.MolToSmiles(mol),node_feature='property_prediction')
    #molstd = td.Molecule.from_smiles(Chem.MolToSmiles(mol),atom_feature='property_prediction')
    
    #molstd = td.Molecule.from_molecule(mol ,node_feature=['property_prediction'])

    compound_node_features = molstd.node_feature # nodes_chemical_features   # 节点特征是分子中每个节点的属性值，例如原子类型、电荷、质量、半径等等
    edge_list = molstd.edge_list # [num_edge, 3]          # 每个元素都是一个元组 (i, j, k)，其中 i 和 j 是分子中两个节点的索引，k 是这两个节点之间的边缘类型。
    edge_weight = molstd.edge_weight # [num_edge, 1]      # edge_weight都等于1


    # 在这段代码中，edge_weight 的最大值和最小值均为 1，因此可以推断出所有边缘的权重都是 1，即所有化学键类型都被视为相同的类型。这种情况通常出现在仅考虑化学键的数量而不是化学键类型的情况下。
    assert edge_weight.max() == 1                         # 检查edge_weight是否都等于1
    assert edge_weight.min() == 1
    assert coords.shape[0] == compound_node_features.shape[0]  # 检查coords变量（未在代码片段中显示）中的行数是否等于分子中节点的数量


    edge_feature = molstd.edge_feature # [num_edge, edge_feature_dim]

    """print("######################################")
    print(edge_feature.shape)
    print("######################################")"""

    x = (coords, compound_node_features, edge_list, edge_feature, pair_dis_distribution)
    return x

import gvp
import gvp.data

three_to_one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 
                'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 
                'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}

def get_clean_res_list(res_list, verbose=False, ensure_ca_exist=False, bfactor_cutoff=None):  # verbose=False, ensure_ca_exist=True
    clean_res_list = []
    for res in res_list:
        hetero, resid, insertion = res.full_id[-1]   # hetero=' ' , resid=1、2、3... , insertion=' '
        if hetero == ' ':  # 执行此条件
            if res.resname not in three_to_one:
                if verbose:    # verbose=False不执行
                    print(res, "has non-standard resname")
                continue
            if (not ensure_ca_exist) or ('CA' in res):    # not ensure_ca_exist = False，不用看  
                                                          # 'CA'属性是指残基的α-碳原子，所以这个函数的作用是去除没有α-碳原子的链
                if bfactor_cutoff is not None:        # bfactor_cutoff = None，不执行
                    ca_bfactor = float(res['CA'].bfactor)
                    if ca_bfactor < bfactor_cutoff:
                        continue
                clean_res_list.append(res)    # 只执行此句
        else:
            if verbose:
                print(res, res.full_id, "is hetero")
    return clean_res_list


def get_protein_feature(res_list):
    # protein feature extraction code from https://github.com/drorlab/gvp-pytorch
    # ensure all res contains N, CA, C and O
    res_list = [res for res in res_list if (('N' in res) and ('CA' in res) and ('C' in res) and ('O' in res))]  # 分别指 氨基端的氮原子、α-碳原子、羧基的碳原子和羧基的氧原子 只有都有时才选取
    # construct the input for ProteinGraphDataset
    # which requires name, seq, and a list of shape N * 4 * 3
    structure = {}
    structure['name'] = "placeholder"
    structure['seq'] = "".join([three_to_one.get(res.resname) for res in res_list])  # 将所有 残基中的氨基酸 按three_to_one表 映射后形成 氨基酸序列：IELTQSPSSLSASLGGKVTITCKASQDIKKYIGWYQHKPGKQPRLL....
    coords = []
    for res in res_list:
        res_coords = []
        for atom in [res['N'], res['CA'], res['C'], res['O']]:   
            res_coords.append(list(atom.coord))     # 获取每个α-碳原子周围四个原子（即残基）的坐标：[[7.752, 36.139, 12.23], [7.101, 36.574, 11.011], [7.217, 35.493, 9.925], [8.231, 34.789, 9.884]]
        coords.append(res_coords)   # 连接成一串
    structure['coords'] = coords
    torch.set_num_threads(1)        # this reduce the overhead, and speed up the process for me.
    dataset = gvp.data.ProteinGraphDataset([structure])       # ！！！对蛋白质结构编码，提取特征！！！  https://github.com/drorlab/gvp-pytorch
    protein = dataset[0]
    x = (protein.x, protein.seq, protein.node_s, protein.node_v, protein.edge_index, protein.edge_s, protein.edge_v)
    """
    x: alpha carbon coordinates
    seq: sequence converted to int tensor according to attribute self.letter_to_num
    name, edge_index
    node_s, node_v: node features as described in the paper with dims (6, 3)
    edge_s, edge_v: edge features as described in the paper with dims (32, 1)
    mask: false for nodes with any nan coordinates
    """
    #print(x)
    return x

# Seed_everything(seed=42)

# used for testing.
def remove_hetero_and_extract_ligand(res_list, verbose=False, ensure_ca_exist=False, bfactor_cutoff=None):
    # get all regular protein residues. and ligand.
    clean_res_list = []
    ligand_list = []
    for res in res_list:
        hetero, resid, insertion = res.full_id[-1]
        if hetero == ' ':
            if (not ensure_ca_exist) or ('CA' in res):
                # in rare case, CA is not exists.
                if bfactor_cutoff is not None:
                    ca_bfactor = float(res['CA'].bfactor)
                    if ca_bfactor < bfactor_cutoff:
                        continue
                clean_res_list.append(res)
        elif hetero == 'W':
            # is water, skipped.
            continue
        else:
            ligand_list.append(res)
            if verbose:
                print(res, res.full_id, "is hetero")
    return clean_res_list, ligand_list

def get_res_unique_id(residue):
    pdb, _, chain, (_, resid, insertion) = residue.full_id
    unique_id = f"{chain}_{resid}_{insertion}"
    return unique_id

def save_cleaned_protein(c, proteinFile):
    res_list = list(c.get_residues())
    clean_res_list, ligand_list = remove_hetero_and_extract_ligand(res_list)
    res_id_list = set([get_res_unique_id(residue) for residue in clean_res_list])

    io=PDBIO()
    class MySelect(Select):
        def accept_residue(self, residue, res_id_list=res_id_list):
            if get_res_unique_id(residue) in res_id_list:
                return True
            else:
                return False
    io.set_structure(c)
    io.save(proteinFile, MySelect())
    return clean_res_list, ligand_list

def split_protein_and_ligand(c, pdb, ligand_seq_id, proteinFile, ligandFile):
    clean_res_list, ligand_list = save_cleaned_protein(c, proteinFile)
    chain = c.id
    # should take a look of this ligand_list to ensure we choose the right ligand.
    seq_id = ligand_seq_id
    # download the ligand in sdf format from rcsb.org. because we pdb format doesn't contain bond information.
    # you could also use openbabel to do this.
    url = f"https://models.rcsb.org/v1/{pdb}/ligand?auth_asym_id={chain}&auth_seq_id={seq_id}&encoding=sdf&filename=ligand.sdf"
    r = requests.get(url)
    open(ligandFile , 'wb').write(r.content)
    return clean_res_list, ligand_list

def generate_conformation(mol):
    mol = Chem.AddHs(mol)
    ps = AllChem.ETKDGv2()
    try:
        rid = AllChem.EmbedMolecule(mol, ps)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500, confId=0)
    except:
        mol.Compute2DCoords()
    mol = Chem.RemoveHs(mol)
    return mol

def write_with_new_coords(mol, new_coords, toFile):
    # put this new coordinates into the sdf file.
    w = Chem.SDWriter(toFile)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        x,y,z = new_coords[i]
        conf.SetAtomPosition(i,Point3D(x,y,z))
    # w.SetKekulize(False)
    w.write(mol)
    w.close()

def generate_sdf_from_smiles_using_rdkit(smiles, rdkitMolFile, shift_dis=30, fast_generation=False):
    mol_from_rdkit = Chem.MolFromSmiles(smiles)
    if fast_generation:
        # conformation generated using Compute2DCoords is very fast, but less accurate.
        mol_from_rdkit.Compute2DCoords()
    else:
        mol_from_rdkit = generate_conformation(mol_from_rdkit)
    coords = mol_from_rdkit.GetConformer().GetPositions()
    new_coords = coords + np.array([shift_dis, shift_dis, shift_dis])
    write_with_new_coords(mol_from_rdkit, new_coords, rdkitMolFile)

def select_chain_within_cutoff_to_ligand_v2(x):
    # pdbFile = f"/pdbbind2020/pdbbind_files/{pdb}/{pdb}_protein.pdb"
    # ligandFile = f"/pdbbind2020/renumber_atom_index_same_as_smiles/{pdb}.sdf"
    # toFile = f"{toFolder}/{pdb}_protein.pdb"
    # cutoff = 10
    pdbFile, ligandFile, cutoff, toFile = x
    
    parser = PDBParser(QUIET=True)
    s = parser.get_structure("x", pdbFile)
    all_res = get_clean_res_list(s.get_residues(), verbose=False, ensure_ca_exist=True)
    all_atoms = [atom for res in all_res for atom in res.get_atoms()]
    protein_coords = np.array([atom.coord for atom in all_atoms])
    chains = np.array([atom.full_id[2] for atom in all_atoms])

    mol = Chem.MolFromMolFile(ligandFile)
    lig_coords = mol.GetConformer().GetPositions()

    protein_atom_to_lig_atom_dis = scipy.spatial.distance.cdist(protein_coords, lig_coords)

    is_in_contact = (protein_atom_to_lig_atom_dis < cutoff).max(axis=1)
    chains_in_contact = set(chains[is_in_contact])
    
    # save protein chains that belong to chains_in_contact
    class MySelect(Select):
        def accept_residue(self, residue, chains_in_contact=chains_in_contact):
            pdb, _, chain, (_, resid, insertion) = residue.full_id
            if chain in chains_in_contact:
                return True
            else:
                return False

    io=PDBIO()
    io.set_structure(s)
    io.save(toFile, MySelect())