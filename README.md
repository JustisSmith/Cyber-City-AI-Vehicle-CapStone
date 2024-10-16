# Cyber-City-AI-Vehicle-CapStone

# Current AI solutions for autonomous vehicles often suffer from poor real-time
# performance and the necessity for multiple boards, which complicates integration and
# reduces overall system efficiency. This project addresses these issues by developing a
# single-board, near-real-time solution using the FPGA fabric of the Xilinx MPSoC to
# accelerate the AI model. Our design aims to provide a compact, efficient, and scalable
# solution that integrates all necessary functionalities—motor control, AI processing, and
# sensor data acquisition—on the Ultra96 board.

# Background:
# The Cyber-City is a small-scale model city designed to facilitate research in smart city
# technologies. It includes an IR motion tracking system (Vicon System) that serves as
# the city’s GPS, enabling precise location tracking. To enhance the Cyber-City's research
# capabilities, there is a need for a small-scale unmanned surface vehicle (USV) that can navigate the city’s lanes, which are constrained in size. A single-board solution is
# optimal for meeting these size requirements while maintaining high performance.

# Objectives:
# - Design and build a remote-controlled (RC) car that meets the size requirements for
# navigating the Cyber-City lanes, utilizing 3D printing for the vehicle's construction.
# - Implement a pre-trained vision AI model on the Xilinx MPSoC for autonomous driving,
# leveraging the FPGA for hardware acceleration.
# - Create circuits for motor and servo control that interface with the MPSoC board,
# ensuring seamless operation within the small-scale vehicle.
# - Integrate position data from the Vicon System via UDP packets into the MPSoC,
# allowing the vehicle to accurately determine its location within the Cyber-City.
# - Develop control logic on the ARM chip within the MPSoC that uses the position data and
# AI model to navigate from point A to point B within the city.
