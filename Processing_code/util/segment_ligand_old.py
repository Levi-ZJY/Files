from rdkit import Chem
from rdkit.Chem import BRICS
import re


def segment_ligand(fileName):

    # 从SDF文件中读取分子
    # proteinName = "1a1e"
    # sdf_file = f"./pdbbind_files_copy/{proteinName}/{proteinName}_ligand.sdf"
    
    print(fileName)

    sdf_reader = Chem.SDMolSupplier(fileName)
    mol = sdf_reader[0]

    smiles = Chem.MolToSmiles(mol)
    print(smiles)

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

        atom_coordinates[atom.GetIdx()] = pos

    # print("原子坐标：", atom_coordinates)

    # 使用BRICS对分子进行切分
    fragments = BRICS.BRICSDecompose(mol)
    fragment_list = list(fragments)


    # 使用正则表达式找到以数字（d）开头，后面跟着一个星号（*）的模式
    pattern = r"\d+\*"

    fragment_list_H = []
    # 将匹配到的模式替换为 "H"
    for string in fragment_list:
        tmp = re.sub(pattern, "H", string)
        fragment_list_H.append(tmp)

    #print(fragment_list_H)

    # 找出切分后各块与原分子的匹配index
    fragment_index = []

    for part in fragment_list_H:
        query_mol = Chem.MolFromSmiles(part)
        match = mol.GetSubstructMatch(query_mol)
        fragment_index.append(list(match))

    print(fragment_index)

    # 获得切分后各个原子的坐标
    fragment_coordinates = {}
    for i,index_list in enumerate(fragment_index):
        coordinates = []
        for index in index_list:
            cod = atom_coordinates[index]
            coordinates.append(cod)
        fragment_coordinates[i] = coordinates

    #for i in fragment_coordinates.keys():
    #    print(i, fragment_coordinates[i])


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

    #for i in barycenter.keys():
    #    print(barycenter[i])

    return barycenter
    

