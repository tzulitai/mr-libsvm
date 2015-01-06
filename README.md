MR-LIBSVM
=========

MR-LIBSVM extends the API of LIBSVM[C.-C Chang,2001], applying the similar API with execution on cluster computer.

MR-LIBSVM have the following additional options:

		MR-LIBSVM included:
		-mr hadoop_uri mapreduce : mapreduce execution mode (NOTE: must provide URI of Hadoop cluster)
		-cc n : n-subset cascade (default 16)

## API format:

Train:

		java -classpath target/mr-libsvm.jar svm_train -mr (hadoop_address) -cc (par_count) data_file


Version Info
----------------

  - Hadoop Version: 2.4.1
  - LIBSVM Version: 3.17

Build Info
----------------
	git clone https://github.com/tzulitai/mr-libsvm.git
	cd mr-libsvm
	mvn clean install

We have used the [LIBSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/) library for the core training procedures: Reference:
> C.-C. Chang and C.-J. Lin. LIBSVM : a library for support vector machines. ACM Transactions on Intelligent Systems and Technology, 2:27:1--27:27, 2011.
