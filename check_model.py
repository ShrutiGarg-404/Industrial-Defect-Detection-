# import torch

# model = torch.load("outputs/models/patchcore_metal_nut_full.pth", map_location="cpu", weights_only=False)

# print("=== Top level attributes ===")
# print([a for a in dir(model) if not a.startswith("_")])

# print()
# print("=== post_processor ===")
# if hasattr(model, "post_processor"):
#     pp = model.post_processor
#     print(type(pp))
#     print([a for a in dir(pp) if not a.startswith("_")])

# print()
# print("=== evaluator ===")
# if hasattr(model, "evaluator"):
#     ev = model.evaluator
#     print(type(ev))
#     print([a for a in dir(ev) if not a.startswith("_")])

# print()
# print("=== pre_processor ===")
# if hasattr(model, "pre_processor"):
#     pre = model.pre_processor
#     print(type(pre))


import torch

categories = ["leather", "tile", "metal_nut", "wood"]

for cat in categories:
    model = torch.load(f"outputs/models/patchcore_{cat}_full.pth", map_location="cpu", weights_only=False)
    pp = model.post_processor
    print(f"{cat}:")
    print(f"  image_threshold: {pp.image_threshold}")
    print(f"  image_min: {pp.image_min}")
    print(f"  image_max: {pp.image_max}")
    print()