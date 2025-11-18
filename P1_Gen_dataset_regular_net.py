# Generate dataset about regular fiber networks
# Yunhao Yang, Fudan University
# --------------------
import os
import copy
import itertools
import json
import random
import numpy as np
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
# --------------------
class SquareGraphGenerator:
    def __init__(self, base_path=os.getcwd()):
        self.base_path = base_path
        self.dataset_path = os.path.join(self.base_path, 'Dataset')
        self.graph_data_path = os.path.join(self.dataset_path, 'Graph_Data')
        self.image_data_path = os.path.join(self.dataset_path, 'Image_Data')
        self.weld_graph_data_path = os.path.join(self.dataset_path, 'Weld_Graph_Data')
        os.makedirs(self.graph_data_path, exist_ok=True)
        os.makedirs(self.image_data_path, exist_ok=True)
        os.makedirs(self.weld_graph_data_path, exist_ok=True)

    def generate_square_graph(self, side_length, num_points_per_side):
        G = nx.Graph()
        vertices = {'A': (0, 0), 'B': (side_length, 0), 'C': (side_length, side_length), 'D': (0, side_length)}
        for vertex, position in vertices.items():
            G.add_node(vertex, pos=position)
        edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')]
        for start, end in edges:
            G.add_edge(start, end)
            start_pos = np.array(vertices[start])
            end_pos = np.array(vertices[end])
            for j in range(1, num_points_per_side + 1):
                t = j / (num_points_per_side + 1)
                point_pos = (1 - t) * start_pos + t * end_pos
                point_name = f"{start}{end}{j}"
                G.add_node(point_name, pos=tuple(point_pos))
                if j == 1:
                    G.add_edge(start, point_name)
                if j == num_points_per_side:
                    G.add_edge(point_name, end)
                if j > 1:
                    prev_point_name = f"{start}{end}{j-1}"
                    G.add_edge(prev_point_name, point_name)
        return G

    def move_AB(self, G, num, dx, dy, side_length):
        new_G = nx.Graph()
        new_G.add_nodes_from(G.nodes(data=True))
        dx, dy = dx * side_length, dy * side_length
        positions = {
            f'AB{num}': (dx, dy),
            f'BC{num}': (-dy, dx),
            f'CD{num}': (-dx, -dy),
            f'DA{num}': (dy, -dx)
        }
        for node, (dx_offset, dy_offset) in positions.items():
            if node in new_G.nodes:
                current_pos = new_G.nodes[node]['pos']
                new_G.nodes[node]['pos'] = (current_pos[0] + dx_offset, current_pos[1] + dy_offset)
        new_G.remove_edges_from(list(new_G.edges))
        node_list = sorted(new_G.nodes)
        for i in range(len(node_list)):
            new_G.add_edge(node_list[i], node_list[(i + 1) % len(node_list)])
        return new_G

    def scale_and_tile_graph(self, new_G, tiling_number, scale_size_num=1, side_length=10):
        scaled_tiled_G = nx.Graph()
        original_positions = nx.get_node_attributes(new_G, 'pos')
        scale_factor = scale_size_num / (side_length * tiling_number)  # 1 / (10 * 6) = 0.016666...
        for i in range(tiling_number):
            for j in range(tiling_number):
                for node, position in original_positions.items():
                    new_x = (position[0] + i * side_length) * scale_factor
                    new_y = (position[1] + j * side_length) * scale_factor
                    scaled_tiled_G.add_node(f"{node}_{i}_{j}", pos=(new_x, new_y))
        for i in range(tiling_number):
            for j in range(tiling_number):
                for edge in new_G.edges():
                    node1, node2 = edge
                    scaled_tiled_G.add_edge(f"{node1}_{i}_{j}", f"{node2}_{i}_{j}")
        return scaled_tiled_G

    def save_graph_to_json(self, G, original_filename, directory='Graph_Data', suffix="_modified"):
        base_name, ext = os.path.splitext(original_filename)
        new_filename = f"{base_name}{suffix}{ext}"
        new_path = os.path.join(self.dataset_path, directory, new_filename)
        graph_data = nx.node_link_data(G)
        with open(new_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=4)

    def visualize_and_save_graph(self, G, save_path):
        pos = nx.get_node_attributes(G, 'pos')
        x_values, y_values = zip(*pos.values())
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        x_range, y_range = x_max - x_min, y_max - y_min
        aspect_ratio = x_range / y_range if y_range != 0 else 1
        plt.figure(figsize=(6, 6)) #plt.figure(figsize=(6, 6))
        nx.draw_networkx_edges(G, pos, width=3)  #width=1
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.xlim(x_min - 0.05 * x_range, x_max + 0.05 * x_range)
        plt.ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def generate_batch(self, num_images=10, tiling_number=6, side_length=10, num_points_per_side=5, scale_size_num=1):
        for i in tqdm(range(num_images), desc="Generating Images"):
            dx_dy_values = [(round(random.uniform(-0.5, 0.5), 2), round(random.uniform(-0.5, 0.5), 2)) for _ in range(5)] #0.5
            tiling_number_int = int(tiling_number)
            save_name = str(tiling_number_int) + '_' + '_'.join([f'{dx}_{dy}' for dx, dy in dx_dy_values])
            save_path_img = os.path.join(self.image_data_path, save_name + '.png')
            save_path_json = os.path.join(self.graph_data_path, save_name + '.json')
            G = self.generate_square_graph(side_length, num_points_per_side)
            for idx, (dx, dy) in enumerate(dx_dy_values, start=1):
                G = self.move_AB(G, idx, dx, dy, side_length)
            scaled_tiled_G = self.scale_and_tile_graph(G, tiling_number, scale_size_num=scale_size_num, side_length=side_length)
            original_filename = f"{save_name}.json"
            self.save_graph_to_json(scaled_tiled_G, original_filename, directory='Graph_Data', suffix="")
            self.visualize_and_save_graph(scaled_tiled_G, save_path_img)

    def intersection_graph(self, original_graph):
        G = copy.deepcopy(original_graph)
        edges = list(G.edges(data=True))
        new_nodes = {}
        new_edges = []
        intersections = {}
        # Iterate over all unique pairs of edges
        for (u1, v1, data1), (u2, v2, data2) in itertools.combinations(edges, 2):
            # Create LineStrings for both edges
            pos_u1 = G.nodes[u1]['pos']
            pos_v1 = G.nodes[v1]['pos']
            line1 = LineString([pos_u1, pos_v1])
            pos_u2 = G.nodes[u2]['pos']
            pos_v2 = G.nodes[v2]['pos']
            line2 = LineString([pos_u2, pos_v2])
            # Check for intersection
            if line1.intersects(line2):
                intersection = line1.intersection(line2)
                if "Point" == intersection.geom_type:
                    ix, iy = intersection.x, intersection.y
                    intersection_node = f'IX_{ix}_{iy}'
                    if intersection_node not in G.nodes:
                        new_nodes[intersection_node] = {'pos': (ix, iy)}
                    intersections.setdefault((u1, v1), []).append((ix, iy))
                    intersections.setdefault((u2, v2), []).append((ix, iy))
        for node, attrs in new_nodes.items():
            G.add_node(node, pos=attrs['pos'])
        for (u, v, data) in tqdm(edges, total=len(edges)):
            points = intersections.get((u, v), [])
            if not points:
                new_edges.append((u, v))
                continue
            pos_u = G.nodes[u]['pos']
            pos_v = G.nodes[v]['pos']
            line = LineString([pos_u, pos_v])
            sorted_points = sorted(points, key=lambda point: line.project(Point(point)))
            prev_node = u
            for point in sorted_points:
                ix, iy = point
                intersection_node = f'IX_{ix}_{iy}'
                new_edges.append((prev_node, intersection_node))
                prev_node = intersection_node
            new_edges.append((prev_node, v))
        G.remove_edges_from(edges)
        G.add_edges_from(new_edges)
        return G

    def Remove_self_join(self, original_graph):
        G = copy.deepcopy(original_graph)
        edges = list(G.edges(data=True))
        new_edges = []
        for (u, v, data) in edges:
            if G.nodes[u]['pos'] != G.nodes[v]['pos']:
                new_edges.append((u, v))
        G.remove_edges_from(edges)
        G.add_edges_from(new_edges)
        return G

    def process_graph_data(self):
        for filename in tqdm(os.listdir(self.graph_data_path), desc="Processing JSON files"):
            if filename.endswith('.json'):
                input_path = os.path.join(self.graph_data_path, filename)
                with open(input_path, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)
                G_original = nx.node_link_graph(graph_data)
                G_intersection = self.intersection_graph(G_original)
                G_intersection = self.Remove_self_join(G_intersection)
                output_path = os.path.join(self.weld_graph_data_path, filename)
                self.save_graph_to_json_custom(G_intersection, filename, directory='Weld_Graph_Data')

    def save_graph_to_json_custom(self, G, filename, directory='Graph_Data'):
        new_path = os.path.join(self.dataset_path, directory, filename)
        graph_data = nx.node_link_data(G)
        with open(new_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    generator = SquareGraphGenerator()
    for tiling_number in range(3, 4):  # 3, 11
        generator.generate_batch(num_images=1000, tiling_number=tiling_number)#1000
    generator.process_graph_data()