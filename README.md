MR-LIBSVM
=========
MR-LIBSVM is a library for accelerated nonlinear SVM with arbitrary kernel functions using cluster computing.

Extended from [LIBSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/), MR-LIBSVM shares the exact same API while using Hadoop MapReduce as its backend computational engine. 

Besides the original LIBSVM options, MR-LIBSVM also includes:

		-mr hadoop_uri mapreduce : mapreduce execution mode (NOTE: must provide URI of Hadoop cluster)
		-cc n : n-subset cascade (default 16)


Version Info
----------------

  - Hadoop Version: 2.4.1
  - LIBSVM Version: 3.17
 
  
Installation and Building
----------------
	git clone https://github.com/tzulitai/mr-libsvm.git	// download code from Github
	cd mr-libsvm						// move to project root directory
	mvn clean install					// build with maven

The compiled jar file is at *target/mr-libsvm.jar*.

Usage
----------------
1. Setup a Hadoop cluster. See [Apache Hadoop](http://hadoop.apache.org/) for installation procedures. 
	
	Suppose the master node is at *master_uri*.
2. Train model on testing data. Suppose cascade subset count of 1024 is desired:

		java -classpath target/mr-libsvm.jar svm_train -mr master_uri -cc 1024 data_file
	The output support vecotrs of each layer and the final model will be in the HDFS path *data_file_out*.

3. Predict with model (the trained model will be downloaded to local filesystem, so this part is identical to LIBSVM):

		java -classpath target/mr-libsvm.jar svm_predict test_file data_file.model output_path 


References
----------------
> C.-C. Chang and C.-J. Lin. LIBSVM : a library for support vector machines. ACM Transactions on Intelligent Systems and Technology, 2:27:1--27:27, 2011.
