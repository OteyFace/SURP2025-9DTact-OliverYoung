# Hardware & Setup Figures

## Overview
This section contains figures demonstrating the hardware design and setup of the enhanced 9DTact tactile sensor system. The figures show the transparent gel modification, camera integration, and overall system architecture.

## Figures

### hardware_overview.png
**Caption**: Complete hardware overview of the enhanced 9DTact tactile sensor system. (a) 9D-Tact schematic highlighting the optical path and numbered parts. (b) Original 9D-Tact exploded view with matching part numbers. (c) Prototype enclosure side view showing the transparent + translucent gel stack on an acrylic window. (d) Prototype full view with camera board and flexible flat cable (FFC) connection.

**Technical Details**:
- **Modification**: Removed opaque black gel (part 12) and LED board (part 5)
- **Enhancement**: Added transparent (part 10) + translucent (part 1) silicone stack
- **Benefit**: Enables RGB frame capture while preserving calibrated depth sensing
- **Compatibility**: Maintains Pixel→Depth LUT pipeline functionality

**Key Components**:
- **1**: Translucent silicone layer
- **2**: Sensor base
- **4**: Camera
- **6**: Isolation ring
- **8**: Sensor shell
- **9**: Acrylic window
- **10**: Transparent silicone layer
- **12**: (Removed) black gel layer
- **5**: (Removed) LED board

**Research Context**: This hardware modification enables the sensor to function as both a camera and a tactile sensor, supporting the enhanced contact detection and 3D reconstruction capabilities demonstrated in the poster.
