target_depth = 0.12345678
print(type(target_depth),target_depth)
a = "{:.2f} cm".format(100 * target_depth)
b = round(100 * target_depth,2)

print(type(a),a)
print(type(b),b)