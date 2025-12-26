
import numpy as np
import networkx as nx
from sklearn.neighbors import KernelDensity, NearestNeighbors
import pandas as pd

import numpy as np
import networkx as nx
from sklearn.neighbors import KernelDensity, NearestNeighbors

class FACE_KDE:
    def __init__(self, model, data, labels, epsilon=0.6, tp=0.5, td=0.01, constraints=None):
        """
        Implementação baseada no Algoritmo 1 (KDE).
        """
        self.model = model
        self.data = np.array(data)
        self.labels = np.array(labels)
        self.epsilon = epsilon
        self.constraints = constraints
        self.tp = tp  # Threshold de probabilidade
        self.td = td  # Threshold de densidade
        
        # 1. Filtro de Fidelidade: Usamos apenas onde o modelo acerta
        # Isso garante que o grafo seja construído sobre 'casos reais'
        preds = self.model.predict(self.data)
        valid_idx = np.where(preds == self.labels)[0]
        self.valid_data = self.data[valid_idx]
        self.valid_labels = self.labels[valid_idx]
        
        # 2. Estimador de Densidade (p_hat) treinado nos dados válidos
        # O Algoritmo 1 exige um estimador de densidade (p_hat)
        self.kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(self.valid_data)
        
        # 3. Construção do Grafo (Linhas 1-7 do Algoritmo 1)
        self.graph = self._build_graph_kde()

    def _get_density(self, x):
        """Retorna a densidade estimada p_hat(x)."""
        return np.exp(self.kde.score_samples(x.reshape(1, -1)))[0]

    def _build_graph_kde(self):
        """Constrói o grafo respeitando epsilon e restrições."""
        G = nx.DiGraph() if self.constraints else nx.Graph()
        
        # Buscamos vizinhos dentro do raio epsilon
        nn = NearestNeighbors(radius=self.epsilon)
        nn.fit(self.valid_data)
        adj_matrix = nn.radius_neighbors_graph(self.valid_data, self.epsilon, mode='distance')
        
        rows, cols = adj_matrix.nonzero()
        for i, j in zip(rows, cols):
            if i >= j and not isinstance(G, nx.DiGraph): continue
            
            xi, xj = self.valid_data[i], self.valid_data[j]
            dist = adj_matrix[i, j]
            
            # Linha 2: d(xi, xj) <= epsilon e c(xi, xj) é True
            if self._check_constraints(xi, xj):
                mid_point = (xi + xj) / 2
                density_mid = self._get_density(mid_point)
                
                # Linha 7: w_ij = d(xi, xj) / p_hat(mid)
                weight = dist / (density_mid + 1e-9)
                G.add_edge(i, j, weight=weight)
        return G

    def _check_constraints(self, start, end):
        if self.constraints is None: return True
        for idx, direction in self.constraints.items():
            if direction == "increasing" and end[idx] < start[idx]: return False
            if direction == "decreasing" and end[idx] > start[idx]: return False
        return True

    def explain(self, query_instance, target_class=0):
        """
        Busca contrafatual seguindo o Algoritmo 1 de forma robusta.
        """
        # 1. Validação de Alvos Candidatos (I_CT)
        # Filtro: Probabilidade >= tp E Densidade >= td
        probs = self.model.predict_proba(self.valid_data)[:, target_class]
        densities = np.exp(self.kde.score_samples(self.valid_data))
        
        i_ct = np.where((probs >= self.tp) & (densities >= self.td))[0]
        
        if len(i_ct) == 0:
            print(f"DEBUG: Nenhum alvo encontrado com P >= {self.tp} e Densidade >= {self.td:.2e}")
            return None

        # 2. Preparação do Grafo Temporário
        temp_graph = self.graph.copy()
        q_id = "QUERY"
        query_instance_flat = query_instance.flatten()
        
        # 3. Conexão da Query ao Grafo
        nn = NearestNeighbors(radius=self.epsilon)
        nn.fit(self.valid_data)
        dists, idxs = nn.radius_neighbors(query_instance_flat.reshape(1, -1))
        
        connected_edges = 0
        for d, idx in zip(dists[0], idxs[0]):
            # Validação de restrições (c(xi, xj) is True)
            if self._check_constraints(query_instance_flat, self.valid_data[idx]):
                # Cálculo do peso w_ij usando densidade do ponto médio
                mid = (query_instance_flat + self.valid_data[idx]) / 2
                density_mid = self._get_density(mid)
                
                # Peso = d(xi, xj) / p_hat(mid)
                w = d / (density_mid + 1e-9)
                temp_graph.add_node(q_id) # Garante que o nó existe
                temp_graph.add_edge(q_id, idx, weight=w)
                connected_edges += 1

        # 4. Verificação Crítica de Conectividade Inicial
        if q_id not in temp_graph:
            print(f"DEBUG: Query isolada. Nenhum vizinho em epsilon={self.epsilon} respeita as restrições.")
            return None

        # 5. Busca do Caminho Mais Curto (Dijkstra Otimizado)
        # Em vez de um loop, calculamos todas as distâncias a partir da QUERY uma única vez
        try:
            lengths, paths = nx.single_source_dijkstra(temp_graph, q_id, weight='weight')
            
            best_target = None
            min_weight = float('inf')
            
            # Filtramos apenas os caminhos que levam a um índice em I_CT
            for target_idx in i_ct:
                if target_idx in lengths and lengths[target_idx] < min_weight:
                    min_weight = lengths[target_idx]
                    best_target = target_idx
            
            if best_target is not None:
                path_nodes = paths[best_target]
                # Retorna o array de coordenadas: [Query, Passo1, ..., Alvo]
                return np.array([query_instance_flat] + [self.valid_data[i] for i in path_nodes[1:]])
                
        except Exception as e:
            print(f"DEBUG: Erro durante o cálculo do caminho: {e}")
            
        print("DEBUG: Nenhum caminho conectando a Query aos candidatos do I_CT.")
        return None

class FACE:
    def __init__(self, model, training_data, target_labels, n_neighbors=20, constraints=None, mode='endpoint'):
        self.model = model
        self.data = np.array(training_data)
        self.labels = np.array(target_labels)
        self.n_neighbors = n_neighbors
        self.constraints = constraints
        self.mode = mode
        
        # Filtro de Fidelidade (densidade confiável)
        preds = self.model.predict(self.data)
        valid_idx = np.where(preds == self.labels)[0]
        self.valid_data = self.data[valid_idx]
        self.valid_labels = self.labels[valid_idx]
        
        self.graph = self._build_base_graph()

    def _check_constraints(self, start, end):
        if self.constraints is None: return True
        for idx, direction in self.constraints.items():
            if direction == "increasing" and end[idx] < start[idx]: return False
            if direction == "decreasing" and end[idx] > start[idx]: return False
        return True

    def _build_base_graph(self):
        """Constrói o grafo base com os dados de treino."""
        nn = NearestNeighbors(n_neighbors=self.n_neighbors)
        nn.fit(self.valid_data)
        # Se mode for 'edge', usamos DiGraph para as conexões entre vizinhos
        G = nx.DiGraph() if (self.mode == 'edge' and self.constraints) else nx.Graph()
        
        distances, indices = nn.kneighbors(self.valid_data, return_distance=True)
        for i in range(len(self.valid_data)):
            G.add_node(i, label=self.valid_labels[i])
            for j_idx, dist in enumerate(distances[i]):
                neighbor_idx = indices[i][j_idx]
                if i == neighbor_idx: continue
                
                # No modo 'edge', as arestas internas também são filtradas
                if self.mode == 'edge':
                    if self._check_constraints(self.valid_data[i], self.valid_data[neighbor_idx]):
                        G.add_edge(i, neighbor_idx, weight=dist)
                else:
                    G.add_edge(i, neighbor_idx, weight=dist)
        return G

    def explain(self, query_instance, target_class=0):
        # 1. Injetar a query dinamicamente
        query_node_id = "USER_QUERY"
        temp_graph = self.graph.copy()
        temp_graph.add_node(query_node_id, label=None)
        
        # 2. Conectar a query aos vizinhos que respeitam as restrições
        nn = NearestNeighbors(n_neighbors=self.n_neighbors)
        nn.fit(self.valid_data)
        distances, indices = nn.kneighbors(query_instance.reshape(1, -1))
        
        connected = False
        for dist, idx in zip(distances[0], indices[0]):
            if self._check_constraints(query_instance, self.valid_data[idx]):
                temp_graph.add_edge(query_node_id, idx, weight=dist)
                connected = True
        
        if not connected:
            return None

        # 3. Filtrar alvos válidos (classe correta e restrição de endpoint)
        targets = [n for n, attr in temp_graph.nodes(data=True) 
                   if n != query_node_id and attr['label'] == target_class]
        
        if self.constraints and self.mode == 'endpoint':
            targets = [t for t in targets if self._check_constraints(query_instance, self.valid_data[t])]

        # 4. Busca do caminho mais curto
        best_path = None
        min_dist = float('inf')
        for t in targets[:100]:
            try:
                d = nx.dijkstra_path_length(temp_graph, query_node_id, t, weight='weight')
                if d < min_dist:
                    min_dist = d
                    best_path = nx.dijkstra_path(temp_graph, query_node_id, t, weight='weight')
            except nx.NetworkXNoPath: continue
        
        if best_path:
            # O primeiro nó é 'USER_QUERY', os demais são índices de valid_data
            path_coords = [query_instance]
            for node_idx in best_path[1:]:
                path_coords.append(self.valid_data[node_idx])
            return np.array(path_coords)
        return None
    
def run_face_experiment(explainer, test_scaled, scaler, feature_names, instance_idx=0):
    """
    Executa o FACE para um índice específico e formata os resultados.
    """
    # 1. Selecionar a instância
    query = test_scaled[instance_idx]
    
    print(f"--- Experimento com Instância Index {instance_idx} ---")
    
    # 2. Gerar o caminho no grafo
    path_scaled = explainer.explain(query, target_class=0)
    
    if path_scaled is None:
        print("Resultado: Nenhum caminho encontrado dentro das restrições de densidade.")
        return None, None # Retorne dois Nones para manter a consistência

    # 3. Transformação inversa e criação de DataFrame
    path_original = scaler.inverse_transform(path_scaled)
    df_steps = pd.DataFrame(path_original, columns=feature_names)
    
    # 4. Cálculo de Mudanças (Delta)
    start_state = df_steps.iloc[0]
    end_state = df_steps.iloc[-1]
    delta = end_state - start_state
    
    # Criar tabela de resumo apenas com o que mudou
    summary = pd.DataFrame({
        'Diferença (Δ)': delta
    })
    
    # Filtrar apenas as linhas onde houve mudança (Δ != 0)
    summary_filtered = summary[summary['Diferença (Δ)'] != 0]
    
    print(f"Sucesso! Caminho encontrado com {len(df_steps)} passos.")
    print(summary_filtered)
    
    return df_steps, path_scaled

import pandas as pd

def get_path_probabilities(model, path_scaled):
    """
    Recebe o caminho escalonado e retorna as probabilidades 
    de predição para cada passo do contrafatual.
    """
    if path_scaled is None:
        return "Nenhum caminho fornecido."
    
    # Obtém as probabilidades para todos os pontos do caminho de uma vez
    probas = model.predict_proba(path_scaled)
    
    # Cria um DataFrame para facilitar a visualização
    # Assumindo: Classe 0 = Aprovado (Good), Classe 1 = Risco (At Risk)
    df_probas = pd.DataFrame(probas, columns=['Prob_Aprovado (0)', 'Prob_Risco (1)'])
    df_probas.index.name = 'Passo'
    df_probas.reset_index(inplace=True)
    
    return df_probas