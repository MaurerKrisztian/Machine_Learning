## Predict Mass Index based on weight and height

Setup:
- Install dependencies
- Train the model

```shell
python3 train_model.py
```
This will train the model on `data/peoples.csv` and save it to `/model/my_model.pkl`

- apply to new data

```python
def main():
    model = utils.load_model("./model/my_model.pkl")
    height=177
    weight=76
    make_predictions(model,  height, weight)
```

```shell
python3 predict.py
```


