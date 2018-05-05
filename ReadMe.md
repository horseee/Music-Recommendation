## README

### Requirement

整个项目在python3.6.5下完成，需要依赖于以下几个包  

* tensorflow
* numpy
* scipy

### 数据

因为github上面无法上传大于100M的文件，源数据无法上传。可以在以下两个地方下载：

* [MSD challenge]
* [Million Song Dataset]

### 运行指令

	python3 preprocessed.py
	python3 main.py 

### 注意点
* 如果使用CPU，该推荐系统对内存的要求较高。在该版本中已将ALS中K的参数调到了500。如果要达到最佳的效果，该值需要在800以上。(在main.py的第112行)
* 如果使用GPU，修改以下几处：去掉140、141、192行的注释，142、193行加上注释。默认CPU。


[MSD challenge]:https://www.kaggle.com/c/msdchallenge
[Million Song Dataset]:https://labrosa.ee.columbia.edu/millionsong/