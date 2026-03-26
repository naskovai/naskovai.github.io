# Section 1. Visualizing Loss Landscapes

A useful analogy for thinking about loss landscapes is to imagine we are hiking across terrain. Consider three types:

## Isotropy

[Image 1: Isotropic terrain plot (circular bowl)]

You're walking down into a perfectly round crater. Every direction feels the same: the slope toward the center is identical no matter which way you turn. You just walk straight down toward the center. From above, the terrain has circular contours: perfect circles around the center point.

## Axis-aligned anisotropy

[Image 2: Axis-aligned anisotropic terrain plot (stretched trough)]

You're on a hillside. In one direction the slope is very steep, while in the perpendicular direction the slope is gentle. You can walk for miles in the gentle direction with barely any elevation change, but a few steps in the steep direction and you're tumbling downhill. These steep and gentle directions align perfectly with the coordinate directions. From above, the terrain has elliptical contours: stretched ovals aligned with the coordinate grid.

## General anisotropy

[Image 3: Rosenbrock-like terrain with curved path]

You're walking down a curved banana-like path that descends through the terrain. From above, the terrain has curved contours that twist and don't align with any coordinate grid.

## How This Maps to LLM Layers

**Axis-aligned anisotropic terrain:**
- Token embeddings (input and output/LM head)
- LayerNorm / RMSNorm scale and bias parameters
- Bias terms

These parameters are mostly independent per dimension. Each embedding dimension captures different semantic features, and LayerNorm scales work per-channel. The loss landscape has steep and gentle directions that align with the parameter coordinates.

**General anisotropic terrain:**
- Attention projection matrices ($Q$, $K$, $V$, $W_o$)
- MLP weight matrices ($W_1$, $W_2$)

These are dense matrices that strongly mix dimensions. When you change one entry in a Q matrix, it affects how that token attends to all other tokens, creating complex interdependencies. The resulting loss landscape has curved valleys that don't align with the parameter grid.
