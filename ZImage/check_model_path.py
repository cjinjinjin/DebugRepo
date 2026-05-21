import os
for d in ["/Model", "/Model/model", "/model", "/models"]:
    if os.path.exists(d):
        print(f"{d} exists:")
        try:
            print("  ", os.listdir(d)[:20])
        except:
            print("  (cannot list)")
    else:
        print(f"{d} NOT found")

# Also check env vars
for k, v in os.environ.items():
    if "model" in k.lower() or "path" in k.lower() or "zimage" in k.lower():
        print(f"ENV {k}={v}")
