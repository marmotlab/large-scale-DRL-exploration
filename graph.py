# Adapted from https://gist.github.com/betandr/541a1f6466b6855471de5ca30b74cb31

class Edge:
    def __init__(self, to_node, length):
        self.to_node = to_node
        self.length = length


class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = dict()

    def add_node(self, node):
        self.nodes.add(node)

    def add_edge(self, from_node, to_node, length):
        edge = Edge(to_node, length)
        # edge = to_node
        if from_node in self.edges:
            from_node_edges = self.edges[from_node]
        else:
            self.edges[from_node] = dict()
            from_node_edges = self.edges[from_node]

        from_node_edges[to_node] = edge

    def clear_edge(self, from_node):
        if from_node in self.edges:
            self.edges[from_node] = dict()


def h(index, destination, node_coords):
    current = node_coords[index]
    end = node_coords[destination]
    h = abs(end[0] - current[0]) + abs(end[1] - current[1])
    # h = ((end[0]-current[0])**2 + (end[1] - current[1])**2)**(1/2)
    return h


def a_star(start, destination, node_coords, graph):
    if start == destination:
        return [], 0
    if str(destination) in graph.edges[str(start)].keys():
        cost = graph.edges[str(start)][str(destination)].length
        return [start, destination], cost
    open_list = {start}
    closed_list = set([])

    g = {start: 0}
    parents = {start: start}

    while len(open_list) > 0:
        n = None
        h_n = 1e5
        # print('open list', open_list)
        for v in open_list:
            h_v = h(v, destination, node_coords)
            if n is not None:
                h_n = h(n, destination, node_coords)
            if n is None or g[v] + h_v < g[n] + h_n:
                n = v

        if n is None:
            print('Path does not exist!')
            return None, 1e5

        if n == destination:
            reconst_path = []
            while parents[n] != n:
                reconst_path.append(n)
                n = parents[n]
            reconst_path.append(start)
            reconst_path.reverse()
            # print('Path found: {}'.format(reconst_path))
            # print(g[destination])
            return reconst_path, g[destination]

        for edge in graph.edges[str(n)].values():
            m = int(edge.to_node)
            cost = edge.length
            # print(m, cost)
            if m not in open_list and m not in closed_list:
                open_list.add(m)
                parents[m] = n
                g[m] = g[n] + cost

            else:
                if g[m] > g[n] + cost:
                    g[m] = g[n] + cost
                    parents[m] = n

                    if m in closed_list:
                        closed_list.remove(m)
                        open_list.add(m)

        open_list.remove(n)
        closed_list.add(n)

    print('Path does not exist!')
    return None, 1e5



