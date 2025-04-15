import os
from pathlib import Path
from typing import Dict, List, Tuple, Union, cast

import cairo
import cairosvg  # type: ignore
from rdkit import Chem  # type: ignore
from rdkit.Chem import Draw  # type: ignore

FilteredDict = Dict[str, Union[str, List["FilteredDict"]]]


class RetroSynthesisTree:
    def __init__(self, idx: int = 0) -> None:
        self.node_id = idx
        self.smiles: str
        self.children: List[RetroSynthesisTree] = []

    def build_tree(self, path_dict: FilteredDict) -> int:
        self.smiles = cast(str, path_dict["smiles"])
        cur_id = self.node_id
        cur_id += 1
        if "children" in path_dict:
            for child in cast(FilteredDict, path_dict["children"]):
                node = RetroSynthesisTree(idx=cur_id)
                cur_id = node.build_tree(path_dict=cast(FilteredDict, child))
                self.children.append(node)
        return cur_id

    def __str__(self) -> str:
        child_ids = [child.node_id for child in self.children]
        header = (
            f"Node ID: {self.node_id}, Children: {child_ids}, SMILES: {self.smiles}\n"
        )
        body = ""
        for child in self.children:
            body += str(child)
        return header + body
    


def draw_rounded_rectangle(
    ctx: cairo.Context,  # type: ignore
    x: int,
    y: int,
    width: int,
    height: int,
    corner_radius: int,
) -> None:
    """Draws a rounded rectangle."""
    ctx.new_sub_path()
    ctx.arc(
        x + width - corner_radius, y + corner_radius, corner_radius, -0.5 * 3.14159, 0
    )
    ctx.arc(
        x + width - corner_radius,
        y + height - corner_radius,
        corner_radius,
        0,
        0.5 * 3.14159,
    )
    ctx.arc(
        x + corner_radius,
        y + height - corner_radius,
        corner_radius,
        0.5 * 3.14159,
        3.14159,
    )
    ctx.arc(x + corner_radius, y + corner_radius, corner_radius, 3.14159, 1.5 * 3.14159)
    ctx.close_path()
    
def compute_subtree_dimensions(
    tree: "RetroSynthesisTree", 
    img_width: int, 
    img_height: int, 
    x_margin: int,
    y_margin: int,
    padding: int = 60  # Increased padding
) -> Tuple[int, int]:
    """Compute the dimensions of the subtree rooted at the given node."""
    # Add padding to basic dimensions
    node_width = img_width + 2 * padding
    node_height = img_height + 2 * padding

    if not tree.children:
        return node_width, node_height

    total_width = -x_margin
    max_child_height = 0
    
    for child in tree.children:
        child_width, child_height = compute_subtree_dimensions(
            child, img_width, img_height, x_margin, y_margin, padding
        )
        total_width += child_width + x_margin
        max_child_height = max(max_child_height, child_height)
    
    return max(total_width, node_width), max_child_height + node_height + y_margin

def draw_molecule_tree(
    tree: "RetroSynthesisTree",
    filename: str,
    width: int = 400,
    height: int = 400,
    x_margin: int = 200,
    y_margin: int = 250,
    padding: int = 60,
    canvas_margin: int = 100  # Added extra canvas margin
) -> None:
    def get_tree_width(node: "RetroSynthesisTree") -> int:
        """Calculate width needed for a subtree"""
        if not node.children:
            return width
        
        children_width = sum(get_tree_width(child) for child in node.children)
        spacing_width = (len(node.children) - 1) * x_margin
        return max(width, children_width + spacing_width)

    def get_tree_layout(
        node: "RetroSynthesisTree",
        x: float,
        y: float,
        available_width: float
    ) -> Dict[int, Tuple[float, float]]:
        """Calculate positions for all nodes in the tree"""
        positions = {node.node_id: (x, y)}
        
        if node.children:
            n_children = len(node.children)
            children_widths = [get_tree_width(child) for child in node.children]
            total_width = sum(children_widths) + (n_children - 1) * x_margin
            
            # Calculate starting x position for first child
            start_x = x - total_width/2
            
            # Position each child
            current_x = start_x
            for child, child_width in zip(node.children, children_widths):
                # Center child under its portion of the parent
                child_x = current_x + child_width/2
                child_positions = get_tree_layout(
                    child,
                    child_x,
                    y + height + y_margin,
                    child_width
                )
                positions.update(child_positions)
                current_x += child_width + x_margin
                
        return positions

    def draw_node(
        ctx: cairo.Context,
        node: "RetroSynthesisTree",
        x: float,
        y: float
    ) -> None:
        """Draw a single node (molecule)"""
        try:
            # Calculate node position with padding
            node_x = x - width/2
            
            # Draw white background
            ctx.set_source_rgb(1, 1, 1)
            draw_rounded_rectangle(ctx, node_x, y, width, height, 20)
            ctx.fill()
            
            mol = Chem.MolFromSmiles(node.smiles)
            if mol:
                # Ensure 2D coordinates
                if mol.GetNumConformers() == 0:
                    Chem.rdDepictor.Compute2DCoords(mol)
                
                # Draw with larger size for better quality
                img = Draw.MolToImage(mol, size=(width * 2, height * 2))
                temp_file = f"temp_mol_{node.node_id}.png"
                img.save(temp_file)
                
                img_surface = cairo.ImageSurface.create_from_png(temp_file)
                
                # Calculate scaling
                scale_x = (width - padding) / img_surface.get_width()
                scale_y = (height - padding) / img_surface.get_height()
                scale = min(scale_x, scale_y)
                
                # Center the image
                img_width = img_surface.get_width() * scale
                img_height = img_surface.get_height() * scale
                img_x = node_x + (width - img_width)/2
                img_y = y + (height - img_height)/2
                
                ctx.save()
                ctx.translate(img_x, img_y)
                ctx.scale(scale, scale)
                ctx.set_source_surface(img_surface, 0, 0)
                ctx.paint()
                ctx.restore()
                
                os.remove(temp_file)
            
            # Draw border
            ctx.set_source_rgb(0, 0, 1)
            ctx.set_line_width(2)
            draw_rounded_rectangle(ctx, node_x, y, width, height, 20)
            ctx.stroke()
            
            # Draw ID
            ctx.set_source_rgb(0, 0, 0)
            ctx.select_font_face("Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
            ctx.set_font_size(20)
            id_text = f"ID: {node.node_id}"
            text_extents = ctx.text_extents(id_text)
            text_x = x - text_extents.width/2
            ctx.move_to(text_x, y + height + 30)
            ctx.show_text(id_text)
            
        except Exception as e:
            print(f"Error drawing node {node.node_id}: {e}")

    # Calculate initial tree width
    total_width = get_tree_width(tree)
    
    # Add extra margins to ensure nothing is cropped
    canvas_width = total_width + 2 * canvas_margin
    
    # Get node positions with adjusted center point
    positions = get_tree_layout(tree, canvas_width/2, y_margin, total_width)
    
    # Calculate total height
    max_y = max(y for _, y in positions.values())
    total_height = max_y + height + y_margin

    # Create surface with extra margins
    surface = cairo.SVGSurface(filename, canvas_width, total_height + y_margin)
    ctx = cairo.Context(surface)
    
    # Set white background
    ctx.set_source_rgb(1, 1, 1)
    ctx.paint()
    
    # Draw all nodes
    for node_id, (x, y) in positions.items():
        # Find node with this ID
        current_node = tree
        stack = [tree]
        while stack:
            node = stack.pop()
            if node.node_id == node_id:
                current_node = node
                break
            stack.extend(node.children)
        
        draw_node(ctx, current_node, x, y)
        
        # Draw connections to children
        if current_node.children:
            for child in current_node.children:
                child_x, child_y = positions[child.node_id]
                ctx.set_source_rgb(0, 0, 0)
                ctx.set_line_width(2)
                ctx.move_to(x, y + height)
                ctx.line_to(child_x, child_y)
                ctx.stroke()
    
    surface.finish()





def create_tree_from_path_string(path_string: str) -> RetroSynthesisTree:
    path_string = path_string.replace('\\N', '\\\\N')
    path_string = path_string.replace('\\n', '\\\\n')
    path_dict: FilteredDict = eval(path_string)
    retro_tree = RetroSynthesisTree()
    retro_tree.build_tree(path_dict=path_dict)
    # print(retro_tree)
    return retro_tree



def draw_tree_from_path_string(
    path_string: str,
    save_path: Path,
    width: int = 400,
    height: int = 400,
    x_margin: int = 200,
    y_margin: int = 250,
) -> None:
    """Create a tree visualization from a path string"""
    assert save_path.suffix == "", "Please provide a path without extension"
    retro_tree = create_tree_from_path_string(path_string=path_string)

    draw_molecule_tree(
        retro_tree,
        filename=str(save_path.with_suffix(".svg")),
        width=width,
        height=height,
        x_margin=x_margin,
        y_margin=y_margin
    )
    
    # Convert SVG to PNG
    svg_path = save_path.with_suffix(".svg")
    png_path = svg_path.with_suffix('.png')
    
    cairosvg.svg2png(
        url=str(svg_path),
        write_to=str(png_path)
    )
    os.remove(svg_path)
    