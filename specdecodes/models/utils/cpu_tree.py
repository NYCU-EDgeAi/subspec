import torch
from typing import Tuple, Dict, List, Optional

class TreeNode:
    __slots__ = (
        'parent',
        'children',
        'depth',
        'token_id',
        'cumulative_probability',
        'has_been_sampled'
    )

    def __init__(
        self,
        parent: Optional[int],
        token_id: int,
        cumulative_probability: float,
        depth: int
    ):
        self.parent = parent
        self.children: List[int] = []
        self.depth = depth
        self.token_id = token_id
        self.cumulative_probability = cumulative_probability
        self.has_been_sampled = False
        
    def __repr__(self):
        return f"TreeNode(token_id={self.token_id}, cumulative_probability={self.cumulative_probability:.4f}, depth={self.depth}, parent={self.parent})"


class Tree:
    """
    CPU-based tree structure with linked (parent-child) nodes.

    Provides methods to add new nodes, prune the tree, retrieve data,
    and create an attention mask based on ancestor relationships.
    """
    
    __slots__ = (
        'prob_dtype',
        'nodes',
        'current_size',
        'available_leaves',
    )

    def __init__(
        self, 
        root_token_id: torch.Tensor,
        prob_dtype: torch.dtype = torch.float32,
    ):
        self.prob_dtype = prob_dtype
        self.nodes: List[TreeNode] = []

        # Create root node
        root_token_id = root_token_id.item()
        root = TreeNode(
            parent=None,
            token_id=root_token_id,
            cumulative_probability=1.0,
            depth=0,
        )
        self.nodes.append(root)
        
        self.current_size = 1
        self.available_leaves: List[int] = [0]

    # Add nodes to the tree once, in a batched manner
    def add_nodes(
        self, 
        token_ids: torch.Tensor,    # shape: [1, total_depth, num_samples]
        token_probs: torch.Tensor,  # shape: [1, total_depth, num_samples]
        local_parent_indices: torch.Tensor,  # shape: [1, total_depth, num_samples]
    ):
        batch_size, total_depth, num_samples = token_ids.shape
        assert batch_size == 1, "Currently only batch_size=1 is supported."

        # Convert data to cpu and list
        local_parent_indices = local_parent_indices.to('cpu', non_blocking=False)
        token_ids = token_ids.to('cpu', non_blocking=False)
        token_probs = token_probs.to('cpu', non_blocking=False)

        local_parent_indices = local_parent_indices.tolist()
        token_ids = token_ids.tolist()
        token_probs = token_probs.tolist()

        for d in range(total_depth):
            # Mark current leaves as sampled
            for leaf_idx in self.available_leaves:
                self.nodes[leaf_idx].has_been_sampled = True
                
            p_inds = local_parent_indices[0][d]
            t_ids = token_ids[0][d]
            probs = token_probs[0][d]

            new_nodes = []
            new_leaves = []
            old_size = self.current_size

            # Create new nodes
            for i, (p_idx, t_id, pr) in enumerate(zip(p_inds, t_ids, probs)):
                parent_idx = self.available_leaves[p_idx]
                parent_node = self.nodes[parent_idx]
                node = TreeNode(
                    parent=parent_idx,
                    token_id=t_id,
                    cumulative_probability=pr,
                    depth=parent_node.depth + 1,
                )
                parent_node.children.append(old_size + i)
                new_leaves.append(old_size + i)
                new_nodes.append(node)

            # Add to the tree and update leaves
            self.nodes.extend(new_nodes)
            self.current_size += len(new_nodes)
            self.available_leaves = new_leaves

    def prune_to_top_n(self, n: int) -> torch.Tensor:
        if n == -1 or self.current_size <= n:
            return torch.arange(self.current_size, device='cpu')

        probs = torch.tensor(
            [node.cumulative_probability for node in self.nodes],
            dtype=self.prob_dtype
        )
        _, keep_idx = torch.topk(probs, k=n, sorted=True)
        keep_list = keep_idx.tolist()

        # Rebuild the nodes
        old2new = {old_i: new_i for new_i, old_i in enumerate(keep_list)}
        new_nodes = []
        for old_i in keep_list:
            o_node = self.nodes[old_i]
            # parent might or might not be in keep_list:
            new_parent = old2new[o_node.parent] if o_node.parent in old2new else None
            n_node = TreeNode(
                parent=new_parent,
                token_id=o_node.token_id,
                cumulative_probability=o_node.cumulative_probability,
                depth=o_node.depth
            )
            n_node.has_been_sampled = o_node.has_been_sampled
            new_nodes.append(n_node)

        # Children references
        for new_i, old_i in enumerate(keep_list):
            for c in self.nodes[old_i].children:
                if c in old2new:
                    new_nodes[new_i].children.append(old2new[c])

        self.nodes = new_nodes
        self.current_size = len(new_nodes)
        is_parent = [bool(n.children) for n in self.nodes]
        self.available_leaves = [
            i for i, node in enumerate(self.nodes)
            if (not is_parent[i]) and (not node.has_been_sampled)
        ]
        return torch.tensor(keep_list, dtype=torch.long)
    
    def get_node(self, node_index: int) -> TreeNode:
        if node_index < 0 or node_index >= self.current_size:
            raise IndexError(f"Node index {node_index} out of bounds for tree size {self.current_size}.")
        return self.nodes[node_index]

    def get_children_indices(self, node_index: int) -> torch.Tensor:
        return torch.tensor(self.nodes[node_index].children, dtype=torch.long, device='cpu')
    
    def get_children_ids(self, node_index: int) -> torch.Tensor:
        return torch.tensor(
            [self.nodes[c].token_id for c in self.nodes[node_index].children],
            dtype=torch.long,
            device='cpu'
        )
    
    def find_child_index(self, node_index: int, match_token_id: int) -> Optional[int]:
        for child_index in self.nodes[node_index].children:
            if self.nodes[child_index].token_id == match_token_id:
                return child_index
        return -1

    def get_tree_data(self, skip_nodes=0) -> Dict[str, torch.Tensor]:
        t_ids, probs, depths, parents = [], [], [], []
        for node in self.nodes:
            t_ids.append(node.token_id)
            probs.append(node.cumulative_probability)
            depths.append(node.depth)
            parents.append(node.parent if node.parent is not None else -1)
            
        # print("Skip nodes:", skip_nodes)
        # print("\tBefore skip nodes:", t_ids)
        # print("\tAfter skip nodes:", t_ids[skip_nodes:])
        return {
            'token_ids': torch.tensor(t_ids[skip_nodes:], dtype=torch.long, device='cpu'),
            'cumulative_probabilities': torch.tensor(probs[skip_nodes:], dtype=self.prob_dtype, device='cpu'),
            'depths': torch.tensor(depths[skip_nodes:], dtype=torch.long, device='cpu'),
            'parent_indices': torch.tensor(parents[skip_nodes:], dtype=torch.long, device='cpu'),
        }
        
    def get_depth(self) -> torch.Tensor:
        return torch.tensor(
            max((node.depth for node in self.nodes), default=0),
            dtype=torch.long,
            device='cpu'
        )
    
    def size(self) -> int:
        return self.current_size

    def create_attention_mask(self, prefix_length: int = 0, skip_nodes: int = 0, device: str = 'cpu') -> torch.Tensor:
        n = self.current_size
        if n == 0:
            return torch.empty((1, 1, 0, prefix_length), dtype=self.prob_dtype, device=device)

        # Mark ancestors (True = can attend)
        ancestor_matrix = [[False]*n for _ in range(n)]
        for i in range(n):
            ancestor_matrix[i][i] = True
            p = self.nodes[i].parent
            while p is not None:
                ancestor_matrix[i][p] = True
                p = self.nodes[p].parent

        am_tensor = torch.tensor(ancestor_matrix, dtype=torch.bool, device=device)
        if prefix_length > 0:
            prefix = torch.ones((n, prefix_length), dtype=torch.bool, device=device)
            # print("prefix mask shape:", prefix.shape, "am_tensor shape:", am_tensor.shape)
            am_tensor = torch.cat([prefix, am_tensor], dim=1)
            # print("whole_am_tensor shape:", am_tensor.shape)

        # Convert to large negative for masking
        # neg_inf_mask = (~am_tensor).to(self.prob_dtype) * torch.finfo(self.prob_dtype).min
        # return neg_inf_mask.unsqueeze(0).unsqueeze(0)
        am_tensor = am_tensor[skip_nodes:, :]
        return am_tensor.unsqueeze(0).unsqueeze(0)
    
    def print_tree_structure(self, show_token_id: bool = True, show_probability: bool = True, tokenizer=None):
        if not (show_token_id or show_probability):
            raise ValueError("At least one of 'show_token_id' or 'show_probability' must be True.")

        children_list = [[] for _ in range(self.current_size)]
        for i, node in enumerate(self.nodes):
            for c in node.children:
                children_list[i].append(c)
                
        def tokenize(c, tokenizer=None):
            if tokenizer:
                return repr(tokenizer.decode([c]))
            return str(c)

        def recurse(idx: int, prefix: str = ''):
            for i, c_idx in enumerate(children_list[idx]):
                connector = '└── ' if i == len(children_list[idx]) - 1 else '├── '
                child_node = self.nodes[c_idx]
                info = []
                if show_token_id:
                    info.append(tokenize(child_node.token_id, tokenizer))
                if show_probability:
                    info.append(f"({child_node.cumulative_probability:.4f})")
                print(prefix + connector + " ".join(info))
                recurse(c_idx, prefix + ('    ' if i == len(children_list[idx]) - 1 else '│   '))

        root = self.nodes[0]
        root_info = []
        if show_token_id:
            root_info.append(tokenize(root.token_id, tokenizer))
        if show_probability:
            root_info.append(f"({root.cumulative_probability:.4f})")
        print(" ".join(root_info))
        recurse(0)

    def __repr__(self):
        return f"LinkedCPUTree(num_nodes={self.current_size}, device='cpu')"