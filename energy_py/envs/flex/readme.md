#  Flexibility environment
Environment to simulate a flexbile electricity asset - ie a chiller.  Also known as demand side response.

The model simulates the control of a chiller using the flow temperature setpoint.  Increasing the setpoint will reduce consumption, reducing the setpoint increases consumption.

Single dimension discrete action space with three choices
- 0 = no nop
- 1 = increase setpoint
- 2 = decrease setpoint
