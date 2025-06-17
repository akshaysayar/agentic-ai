import logging
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Iterable, List, Literal, Tuple, Union
from uuid import UUID

from lxml import etree

from research_tool_rag.utils.utils import text_hash

logger = logging.getLogger(__name__)


@dataclass
class Node:
    name: str
    hierarchical_name: str
    hierarchical_number: str
    hierarchical_title: str
    number: str
    content: str
    version: str
    id: UUID
    parent: "Node"
    child_nodes: List["Node"] = field(default_factory=list)
    hierarchy: "Hierarchy" = field(init=True, default=None)

    def __repr__(self):
        return f"Node({self.number}-{self.name})"

    @staticmethod
    def _get_title(element: etree.Element) -> Tuple[str, str]:
        number = name = ""
        if (name_elm := element.find("name")) is not None:
            name = " ".join(s.strip() for s in name_elm.xpath("text()")).strip()
        if (num_el := element.find("number")) is not None:
            if num_el.text is not None:
                return (num_el.text.strip(), name)
        return (number, name)

    @staticmethod
    def _get_version(element: etree.Element) -> str:
        if (version := element.find("version")) is not None:
            return version.text.strip() if version.text else ""
        return "1"

    @staticmethod
    def _body_text(element: etree.Element) -> str:
        iter_codetext = chain.from_iterable(
            elem.itertext() for elem in element.iterdescendants("codetext")
        )
        stripped = filter(str.strip, iter_codetext)
        return "".join(stripped)

    @staticmethod
    def is_leaf_node(element: etree.Element) -> bool:
        return False if element.xpath("boolean(code)") else True


@dataclass
class Hierarchy:
    state: str = ""
    law_type: Literal["laws", "regs"] = ""
    title: str = ""
    children: List[Node] = field(default_factory=list)
    path: Union[str, Path] = ""  # Store path if you want
    root_node: Node = field(init=False, default=None)

    def get_node_by_locator(self, locator: str) -> Node:
        """
        Get a node by its hierarchy loc.
        Args:
            locator (str): The locator of the node to find.
            exaample: 0/1/2/3
        Returns:
            Node: The node with the specified locator.
        """
        node: Node = None

        for pos in locator.split("/"):
            if pos.isdigit():
                pos = int(pos)
            if node is None:
                node = self.root_node.child_nodes[pos]
            else:
                node = node.child_nodes[pos]

        return node

    def __post_init__(self):
        if isinstance(self.path, str):
            self.path = Path(self.path)

        if not self.path.exists():
            raise FileNotFoundError(f"Path {self.path} does not exist")

        try:
            self.state, self.law_type = self.path.parts[-10].split("-")
            self.title = self.path.parts[-9].replace("_", " ").strip()
        except (IndexError, ValueError) as e:
            raise ValueError(f"Path format invalid: {self.path}") from e

        if self.law_type not in ("laws", "regs"):
            raise ValueError("law_type must be either 'Laws' or 'Regs'")

    def _depth_first_walk(
        self, node: Union[Node, None] = None
    ) -> Iterable[Union[Node, "Hierarchy"]]:
        if node is None:
            node = self
        yield node
        for child in node.child_nodes:
            if child is None:
                raise ValueError(
                    f"Node {node} has a child node_id reference that does not exist in the repo"
                )
            yield from self._depth_first_walk(child)

    def _build_node(self, element: etree.Element, parent: Node) -> Node:
        number, name = Node._get_title(element)
        version = Node._get_version(element)
        hierarchical_title = (
            f"{parent.hierarchical_title} - {number} {name}".replace("  ", " ")
            .replace("\n", "")
            .strip()
        )

        node = Node(
            name=name,
            hierarchical_name=f"{parent.hierarchical_name} -> {name}".replace("  ", " ")
            .replace("\n", "")
            .strip(),
            number=number,
            hierarchical_number=f"{parent.hierarchical_number} -> {number}".replace("  ", " ")
            .replace("\n", "")
            .strip(),
            hierarchical_title=hierarchical_title,
            content=Node._body_text(element),
            version=version,
            id=UUID(text_hash(f"{hierarchical_title}-v{version}"), version=4),
            parent=parent,
            hierarchy=self,
        )

        if Node.is_leaf_node(element):
            # logger.info(f"Created {node}")
            self.children.append(node)
            return node

        for child in element.findall("code"):
            child_node = self._build_node(child, node)
            node.child_nodes.append(child_node)

        return node

    def build_hierarchy(self) -> Node:
        """
        Build a hierarchy of nodes from the XML structure.
        """
        tree = etree.parse(self.path)
        root_element = tree.find('code[@type="Root"]')
        number, name = Node._get_title(root_element)
        version = Node._get_version(root_element)
        hierarchical_title = f"{number} {name}".replace("  ", " ").replace("\n", "").strip()

        self.root_node = Node(
            name=name,
            hierarchical_name=name,
            number=number,
            hierarchical_number=number,
            hierarchical_title=hierarchical_title,
            content="",
            version=version,
            id=UUID(text_hash(f"{hierarchical_title}-v{version}"), version=4),
            parent=None,
            hierarchy=self,
        )

        for child in root_element.findall("code"):
            child_node = self._build_node(child, self.root_node)
            self.root_node.child_nodes.append(child_node)

        return self.root_node
