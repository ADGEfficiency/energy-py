from energypylinear.battery import BatteryEnv, register_env

__all__ = ["BatteryEnv", "register_env"]

# Register the environment when the package is imported
register_env()

def hello() -> str:
    return "Hello from energypylinear!"