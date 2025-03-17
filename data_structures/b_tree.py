from . import BNode

class BTree:
    def __init__(self, root_offset: int, loader, allocator, invalidator) -> None:
        self.root_offset = root_offset   
        self.load_node = loader          
        self.create_node = allocator     
        self.invalidate_node = invalidator 

        root: BNode = self.load_node(root_offset)
        new_root: BNode = self.create_node(root_offset)
        self.invalidate_node(root_offset)