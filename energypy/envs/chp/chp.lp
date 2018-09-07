\* cost minimization *\
Minimize
OBJ: 16.8913125 blr1 + 6.14285714286 gt1
Subject To
_C1: gt1 - 100 gt1_bin <= 0
_C2: - gt1 + 50 gt1_bin <= 0
_C3: blr1 - 100 blr1_bin <= 0
_C4: - blr1 <= 0
power_balance: 0.1 gt1 + net_power_to_site = 100
steam_balance: blr1 + 0.211435823677 gt1 = 100
Bounds
blr1 <= 100
gt1 <= 100
-100 <= net_power_to_site <= 100
Binaries
blr1_bin
gt1_bin
End
