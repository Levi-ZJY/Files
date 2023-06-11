# Files

## Graph files

- Index, Molecular_to_Motif edges, Motif_to_Subpocket edges (~4MB):
  Baidu Netdisk: [GraphEdges1 (passwd: 2eu9)](https://pan.baidu.com/s/1zOaDJI6uX7DHTk2EP5pQ1g )


- Subpocket_to_Subpocket edges (~25GB):
  Baidu Netdisk: [GraphEdges2 (passwd: xe8r)](https://pan.baidu.com/s/1LSiE3rWMX9rwiJIMLuWbyQ  )
  ```
  [ [node, node, weight],  [node, node, weight],  [node, node, weight], ...... ] Weight = Similarity（-0.4<=Weight <=0）
  ```



## Processing code (partical)
    Segment_Ligand_and_Protein.ipynb: Used for splitting molecules and proteins.
    generate_edges.ipynb: Used for generating edges between molecules and Motifs, as well as between Motifs and Subpockets.

