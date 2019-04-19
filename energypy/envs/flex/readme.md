#  Flexibility environment

Environment to simulate a flexbile electricity asset - ie a chiller.  Also known as demand side response.

The model simulates the control of a chiller using the flow temperature setpoint.  Increasing the setpoint will reduce consumption, reducing the setpoint increases consumption.

Single dimension discrete action space with three choices

- 0 = no nop
- 1 = increase setpoint
- 2 = decrease setpoint

Asset can operate in four dimensions

1. store demand (reducing site electricity consumption)
2. release demand (increasing site electricity consumption)
3. storing supply (increasing site electricity consumption)
4. releasing supply (decreasing site electricity consumption)

Storing and releasing demand is the classic use of demand side response
where the control is based on reducing the asset electricity consumption

Storing and releasing supply is the the inverse - controlling an asset by
increasing electricity consumption - avoiding consuming electricity later
