import numpy as np
from scipy.sparse import coo_matrix


def convert_edge_list(csr_adj_matrix):
    coo_adj_matrix = coo_matrix(csr_adj_matrix)
    row = coo_adj_matrix.row
    col = coo_adj_matrix.col
    edge_list = np.zeros((2, len(row))).astype(int)
    edge_list[0, :] = row
    edge_list[1, :] = col
    edge_list = edge_list.transpose()
    return edge_list


def edges_non_edges_index(adj, N, node_neighbors_dict):
    A_coo = coo_matrix(adj)
    A_coo_data = A_coo.data
    diagonal_element = np.array([adj[i, i] for i in range(N)])
    self_loop_index = np.where(diagonal_element == 1)[0]
    self_loop_n = len(self_loop_index)
    links_n = len(np.where(A_coo_data != 0)[0]) - self_loop_n
    links_n = int(links_n/2)
    non_links_n = int((N * N - N)/2 - links_n)

    nonedges_index_a = np.zeros(non_links_n).astype(int)
    nonedges_index_b = np.zeros(non_links_n).astype(int)

    edges_index_a = np.zeros(links_n).astype(int)
    edges_index_b = np.zeros(links_n).astype(int)

    N_list_set = np.array([i for i in range(N)])

    start_edges = 0
    start_non_edges = 0
    for i in range(N):
        # deal with links
        node_i_neighbors = node_neighbors_dict[i]
        node_i_upper_tri_index = np.arange(i+1 , N)
        node_i_neighbors_upper_tri = np.intersect1d(node_i_neighbors, node_i_upper_tri_index)
        end_edges = start_edges + len(node_i_neighbors_upper_tri)
        edges_index_a[start_edges:end_edges] = i
        edges_index_b[start_edges:end_edges] = node_i_neighbors_upper_tri

        start_edges = end_edges

        # deal with non-links
        node_i_non_neighbor = np.setdiff1d(N_list_set, node_i_neighbors)
        node_i_non_neighbor_tri = np.intersect1d(node_i_non_neighbor, node_i_upper_tri_index)
        node_i_non_neighbor_n = len(node_i_non_neighbor_tri)

        end_non_edges = start_non_edges + node_i_non_neighbor_n
        nonedges_index_a[start_non_edges:end_non_edges] = i
        nonedges_index_b[start_non_edges:end_non_edges] = node_i_non_neighbor_tri

        start_non_edges = end_non_edges

    nonedges = (nonedges_index_a, nonedges_index_b)
    edges = (edges_index_a, edges_index_b)
    return edges, nonedges


def step_size_function(itr_index, tao, scaler):
    step_size = (tao + itr_index) ** (-0.5)
    return step_size/scaler
