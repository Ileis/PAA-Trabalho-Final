# PAA-Trabalho-Final

## ['A SIMPLE TEST FOR PLANAR GRAPHS'](https://iasl.iis.sinica.edu.tw/webpdf/paper-1993-A_simple_test_for_planar_graphs.pdf)

- Definicao do problema: Dado um grafo nao-direcionado, o problema do teste de
planaridade consiste em determinar quando um grafo pode ser desenhado em um
plano sem que nenhuma aresta se cruse.

- Algoritmo de Wei-Kuan Shih e Wen-Lian Hsu usa o teorema de Kuratowski.

- 'Nos adotamos a abordagem "adicao de vertices" que somente requere que os
vertices "nao adicionados" induzam um subgrafo conexo'.

- Considerando a ordem dos vertices definida na pos-ordem da arvore de
profundidade de um grafo, vamos considerar Gi sendo o subgrafo da i-esima
iteracao consistindo dos primeiros i vertices e as arestas entre eles. Em
nossa abordagem Gi deve ser desconexo...

- Se um vertice na atual floresta tem um back_edge para i entao deve existir um
tree_edge de i para sua raiz.

- Vamos dizer que as arvores que as raizes sao adjacentes a i na floresta atual
sao T1, T2, ..., Tr.

- Vamos dizer que S eh um subconjunto de vertices. Denotamos como G[S] o
subgrafo de G induzido por S.

- Sendo i um vertice de articulacao do grafo induzido Hi = G[{i} U T1 U T2 ...
U Tr], Hi eh planar se e somente se cada G[{i} U Tj] j = 1, ..., r, for planar.

- Considere o problema de planaridade para cada G[{i} U Tj]. Devemos achar a
"distribuicao plana" de G[{i} U Tj] que corresponde a uma "distribuicao plana"
parcial de G.(rj eh a raiz de Tj).

- Definiremos grau externo de um vertice na iteracao i sendo o numero de
vizinhos em G ainda nao adicionados em Gi.

- Para simplificar nossa descricao nos precisamos aplicar uma "fusao de
vertices, contracao de vertices, reducao de grafo" para eliminar alguns vertices
em Gi-1 com grau externo = 0.

- O processo de contracao comeca marcando todos os vertices de Tj que sao
adjacentes a i. (inducao). Pegue todos os vertices marcados em ordem crescente
(rotulos do posordem)(para pegar a ordem crescente preordem?)...

- Se um vertice marcado eh uma folha com grau externo = 0, entao delete u e
marque o pai de u (estamos dizendo que para ir para u precisamos primeiro ir
para o pai de u, apartir dai ja sabemos que existe um caminho para u).

- Se um vertice u se tornar uma folha, significa que todos os seus decendentes
tem grau externo 0