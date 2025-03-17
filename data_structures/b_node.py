from enum import Enum
import struct

class NodeType(Enum):
    INTERNAL = 1
    LEAF = 2

class BNode:
    """Represents a B-tree node with efficient encoding/decoding logic for KV pairs."""

    HEADER_SIZE = 4
    POINTER_SIZE = 8
    FLAG_SIZE = 2
    CHECKSUM_SIZE = 4
    MAX_KEY_SIZE = 1000
    MAX_VAL_SIZE = 3000
    PAGE_SIZE = 4096

    def __init__(self):
        """Initialize a BNode with a fixed PAGE_SIZE bytearray."""
        total_size = (
            self.HEADER_SIZE +
            self.POINTER_SIZE +
            self.FLAG_SIZE +
            self.CHECKSUM_SIZE +
            self.MAX_KEY_SIZE +
            self.MAX_VAL_SIZE
        )
        assert total_size <= self.PAGE_SIZE, "Node size exceeds page size limit!"
        self.data = bytearray(self.PAGE_SIZE)

    # === NODE TYPE CHECKS ===
    def is_leaf(self) -> bool:
        """Returns True if the node is a leaf."""
        return self.get_node_type() == NodeType.LEAF

    def is_internal(self) -> bool:
        """Returns True if the node is an internal node."""
        return self.get_node_type() == NodeType.INTERNAL

    # === METADATA ===
    def get_node_type(self) -> int:
        """Extracts the 2-byte 'btype' field from the node's data."""
        return struct.unpack_from('<H', self.data, 0)[0]  # Little-endian Uint16 at position 0

    def get_num_keys(self) -> int:
        """Extracts the 2-byte 'nkeys' field from the node's data starting at byte 2."""
        return struct.unpack_from('<H', self.data, 2)[0]

    def write_metadata(self, node_type: int, num_keys: int) -> None:
        """Writes 'node_type' and 'nkeys' back into the node's data buffer."""
        struct.pack_into('<H', self.data, 0, node_type)
        struct.pack_into('<H', self.data, 2, num_keys)

    # === POINTERS ===
    def get_pointer(self, idx: int) -> int:
        """Retrieves a pointer (uint64) from the node's data."""
        assert idx < self.get_num_keys(), "Pointer index out of range."
        pos = self.HEADER_SIZE + (self.POINTER_SIZE * idx)
        return struct.unpack_from('<Q', self.data, pos)[0]

    def set_pointer(self, idx: int, val: int) -> None:
        """Sets a pointer (uint64) in the node's data."""
        assert idx < self.get_num_keys(), "Pointer index out of range."
        pos = self.HEADER_SIZE + (self.POINTER_SIZE * idx)
        struct.pack_into('<Q', self.data, pos, val)

    # === OFFSET LIST ===
    def get_offset_position(self, idx: int) -> int:
        """Calculates the position of the idx-th offset in the byte array."""
        num_keys = self.get_num_keys()
        pos = self.HEADER_SIZE + (num_keys * self.POINTER_SIZE) + 2 * (idx - 1)
        return pos

    def get_offset(self, idx: int) -> int:
        """Retrieves the offset for the idx-th KV pair."""
        assert idx > 0, "Offset index must be greater than zero."
        return struct.unpack_from('<H', self.data, self.get_offset_position(idx))[0]

    def set_offset(self, idx: int, offset: int) -> None:
        """Sets the offset for the idx-th KV pair."""
        pos = self.get_offset_position(idx)
        struct.pack_into('<H', self.data, pos, offset)

    # === KV PAIR MANAGEMENT ===
    def get_kv_start(self, idx: int) -> int:
        """Calculates the starting position of the idx-th KV pair."""
        num_keys = self.get_num_keys()
        assert idx <= num_keys, "KV index out of range."
        return self.HEADER_SIZE + (num_keys * self.POINTER_SIZE) + (2 * num_keys) + self.get_offset(idx)

    def get_key(self, idx: int) -> bytes:
        """Retrieves the key from the idx-th KV pair."""
        assert idx < self.get_num_keys(), "Invalid key index."

        pos = self.get_kv_start(idx)
        key_length = struct.unpack_from('<H', self.data, pos)[0]

        return self.data[pos + 4 : pos + 4 + key_length]

    def get_value(self, idx: int) -> bytes:
        """Retrieves the value from the idx-th KV pair."""
        assert idx < self.get_num_keys(), "Invalid value index."

        pos = self.get_kv_start(idx)
        key_length = struct.unpack_from('<H', self.data, pos)[0]
        value_length = struct.unpack_from('<H', self.data, pos + 2)[0]

        # Guard clause to handle zero-length values
        if value_length == 0:
            return b''

        return self.data[pos + 4 + key_length : pos + 4 + key_length + value_length]

    def get_data_end(self) -> int:
        """Returns the total number of meaningful bytes in the node's data."""
        return self.get_kv_start(self.get_num_keys())  # Last offset is sentinel

