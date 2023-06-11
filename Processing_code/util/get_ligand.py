from rdkit import Chem
from rdkit.Chem import BRICS
import re


def segment_ligand(fileName):

    # 从SDF文件中读取分子
    # proteinName = "1a1e"
    # sdf_file = f"./pdbbind_files_copy/{proteinName}/{proteinName}_ligand.sdf"
    
    #print(fileName)

    sdf_reader = Chem.SDMolSupplier(fileName)
    mol = sdf_reader[0]

    smiles = Chem.MolToSmiles(mol)
    #print(smiles)

    # 获取原子坐标
    conf = mol.GetConformer()

    #atom_coordinates = {atom.GetIdx(): conf.GetAtomPosition(atom.GetIdx()) for atom in mol.GetAtoms()}
    atom_coordinates = {}
    for atom in mol.GetAtoms():         
        posOBJ = conf.GetAtomPosition(atom.GetIdx())
        pos = []
        for i in posOBJ:
            pos.append(i)

        pos.append(atom.GetMass())     # atom.GetMass()用于获取原子对象的相对原子质量。

        atom_coordinates[atom.GetIdx()+1] = pos

    # print("原子坐标：", atom_coordinates)

    # 找到BRICS键
    brics_bonds = BRICS.FindBRICSBonds(mol)

    # 提取键索引
    bond_indices = []
    for bond_info,_ in brics_bonds:
        bond_indices.append(mol.GetBondBetweenAtoms(bond_info[0], bond_info[1]).GetIdx())
    
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx()+1)   # 标记AtomMapNum，设置为该原子的id

    """
    # 对原始分子应用切分
    mol_fragments = Chem.FragmentOnBonds(mol, bond_indices, addDummies=False)
    #print(mol_fragments)

    fragment_list = []
    for i, fragment in enumerate(Chem.GetMolFrags(mol_fragments, asMols=True)):
        #tsmile = Chem.MolToSmiles(fragment)
        #print(tsmile)
        
        atomNum = []
        for atom in fragment.GetAtoms():
            AtomMapNum = atom.GetAtomMapNum()
            atomNum.append(AtomMapNum)

        fragment_list.append(atomNum)
    """

    # 对原始分子应用切分
    fragment_list = []

    if len(bond_indices)!=0:
        #print("cut")
        mol_fragments = Chem.FragmentOnBonds(mol, bond_indices, addDummies=False)

        #print(mol_fragments)

        for i, fragment in enumerate(Chem.GetMolFrags(mol_fragments, asMols=True)):
            #tsmile = Chem.MolToSmiles(fragment)
            #print(tsmile)
            
            atomNum = []
            for atom in fragment.GetAtoms():
                AtomMapNum = atom.GetAtomMapNum()
                atomNum.append(AtomMapNum)

            fragment_list.append(atomNum)
    else:
        #print("nothing to cut")
        atomNum = []
        for atom in mol.GetAtoms():
            AtomMapNum = atom.GetAtomMapNum()
            atomNum.append(AtomMapNum)
        fragment_list.append(atomNum)

    
    fragment_coordinates = {}
    for i,index_list in enumerate(fragment_list):
        coordinates = []
        for index in index_list:
            cod = atom_coordinates[index]
            coordinates.append(cod)
        fragment_coordinates[i] = coordinates

        
    barycenter = {}

    for i in fragment_coordinates.keys():
        cod_list = fragment_coordinates[i]
        tx=0
        ty=0
        tz=0
        tbray=0
        for atom in cod_list:
            tx += atom[0]*atom[3]
            ty += atom[1]*atom[3]
            tz += atom[2]*atom[3]
            tbray += atom[3]
        x = tx/tbray
        y = ty/tbray
        z = tz/tbray
        barycenter[i] = [x,y,z]
    
    return barycenter
    






def get_ligand(fileName):   # 输入ligand的文件路径，输入切分好后的mol对象，以列表形式

    # 从SDF文件中读取分子
    # proteinName = "1a1e"
    # sdf_file = f"./pdbbind_files_copy/{proteinName}/{proteinName}_ligand.sdf"
    
    #print(fileName)

    sdf_reader = Chem.SDMolSupplier(fileName)
    mol = sdf_reader[0]

    smiles = Chem.MolToSmiles(mol)
    #print(smiles)

    # 获取原子坐标
    conf = mol.GetConformer()

    #atom_coordinates = {atom.GetIdx(): conf.GetAtomPosition(atom.GetIdx()) for atom in mol.GetAtoms()}
    atom_coordinates = {}
    for atom in mol.GetAtoms():         
        posOBJ = conf.GetAtomPosition(atom.GetIdx())
        pos = []
        for i in posOBJ:
            pos.append(i)

        pos.append(atom.GetMass())     # atom.GetMass()用于获取原子对象的相对原子质量。

        atom_coordinates[atom.GetIdx()+1] = pos

    # print("原子坐标：", atom_coordinates)

    # 找到BRICS键
    brics_bonds = BRICS.FindBRICSBonds(mol)

    # 提取键索引
    bond_indices = []
    for bond_info,_ in brics_bonds:
        bond_indices.append(mol.GetBondBetweenAtoms(bond_info[0], bond_info[1]).GetIdx())
    
    """
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx()+1)   # 标记AtomMapNum，设置为该原子的id
    """
    # 对原始分子应用切分
    fragment_mol = []

    if len(bond_indices)!=0:
        #print("cut")
        mol_fragments = Chem.FragmentOnBonds(mol, bond_indices, addDummies=False)

        #print(mol_fragments)  
        for i, fragment in enumerate(Chem.GetMolFrags(mol_fragments, asMols=True)):
            #tsmile = Chem.MolToSmiles(fragment)
            #print(tsmile)
            fragment_mol.append(fragment)

    else:
        #print("nothing to cut")
        fragment_mol.append(mol)


    return fragment_mol



from rdkit import Chem
from rdkit.Chem import InchiToInchiKey, MolToSmiles


# 定义一个函数，用于判断两个分子是否相同（基于InChI）
def is_same_molecule_inchi(mol1, mol2):    # 输入两个mol对象，判断两个化学分子是否相同
    # 将两个mol对象转换为InChI字符串
    inchi1 = Chem.MolToInchi(mol1)
    inchi2 = Chem.MolToInchi(mol2)

    #print(inchi1, inchi2)
    
    # 将InChI字符串转换为InChIKey
    inchi_key1 = InchiToInchiKey(inchi1)
    inchi_key2 = InchiToInchiKey(inchi2)

    #print(inchi_key1, inchi_key2)
    
    # 比较两个InChIKey，如果它们相同，则认为这两个分子结构相同
    if inchi_key1 == inchi_key2:
        return True
    else:
        return False



def get_inchikey(mol):    # 输入一个mol对象，返回它的InchiKey
    # 将mol对象转换为InChI字符串
    inchi = Chem.MolToInchi(mol)

    # 将InChI字符串转换为InChIKey
    inchi_key1 = InchiToInchiKey(inchi)

    return inchi_key1






