from model.bda_model import CustomModel

model = CustomModel()
prompt = 'test'
r = model(prompt)
print(r)
