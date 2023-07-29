import subprocess
import os

p = subprocess.Popen(f"kubectl port-forward service/jaeger-out -n social-network --address 0.0.0.0 16686:16686 &", shell=True)
os.system("sleep 60")
p.kill()
