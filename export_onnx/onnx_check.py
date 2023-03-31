import onnx

# load model from onnx

model = onnx.load('./path/to/onnx.onnx')

# confirm model has valid schema
onnx.checker.check_model(model)


# Print a human readable representation of the graph
onnx.helper.printable_graph(model.graph)