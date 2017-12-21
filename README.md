# serving-example

## example.1

require environment
```
python3.x
tensorflow1.x
grpcio1.x
```

run container
```bash
docker build -t serving-example .
docker run --name serving-example -v `pwd`:/root/serving-example -p 8500:8500 -it serving-example /bin/bash
```
start tensorflow model server
```bash
tensorflow_model_server --model_name='default' --model_base_path=/root/serving-example/tmp
```
test example script
```
python example.py --x=2 --y=3 --version=1
```
