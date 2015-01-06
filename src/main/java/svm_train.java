/*
 * This svm_train program is modified from the original LIBSVM's svm_train.
 * Modifications will be pointed out by comments.
 */

import libsvm.*;
import mrlibsvm.MRLibSVM;
import mrlibsvm.MRLibSVM.LastLayerSvmModelOutputReducer;
import mrlibsvm.MRLibSVM.PrePartitionerMapper;
import mrlibsvm.MRLibSVM.PreStatCounterMapper;
import mrlibsvm.MRLibSVM.SubSvmMapper;
import mrlibsvm.MRLibSVM.SubsetDataOutputReducer;
import mrlibsvm.io.TrainingSubsetInputFormat;

import java.io.*;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.filecache.DistributedCache;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import com.google.gson.Gson;

class svm_train {
	private svm_parameter param;		// set by parse_command_line
	private svm_problem prob;		// set by read_problem
	private svm_model model;
	private String input_file_name;		// set by parse_command_line
	private String model_file_name;		// set by parse_command_line
	private String error_msg;
	private int cross_validation;
	private int nr_fold;

	/*
	 * MR-LIBSVM included fields.
	 */
	
	private boolean is_hadoop_mode;		// is hadoop mode enabled?
	private String hadoop_address;		// IP address of hadoop cluster
	private int subset_count;			// subset count for first layer of cascade
	private String output_hdfs_path;	// output path used for storage to HDFS
	private int prepartition_job_count;	// should be 2
	private int cascade_job_count;		// Math.log(subset_count)/Math.log(2)
	private Job[] prepartition_jobs;	// list of Jobs for prepartitioning
	private Job[] cascade_jobs;			// list of Jobs for cascaded training
	
	private static svm_print_interface svm_print_null = new svm_print_interface()
	{
		public void print(String s) {}
	};

	private static void exit_with_help()
	{
		System.out.print(
		 "Usage: svm_train [options] training_set_file [model_file]\n"
		+"options:\n"
		+"-s svm_type : set type of SVM (default 0)\n"
		+"	0 -- C-SVC		(multi-class classification)\n"
		+"	1 -- nu-SVC		(multi-class classification)\n"
		+"	2 -- one-class SVM\n"
		+"	3 -- epsilon-SVR	(regression)\n"
		+"	4 -- nu-SVR		(regression)\n"
		+"-t kernel_type : set type of kernel function (default 2)\n"
		+"	0 -- linear: u'*v\n"
		+"	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
		+"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
		+"	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
		+"	4 -- precomputed kernel (kernel values in training_set_file)\n"
		+"-d degree : set degree in kernel function (default 3)\n"
		+"-g gamma : set gamma in kernel function (default 1/num_features)\n"
		+"-r coef0 : set coef0 in kernel function (default 0)\n"
		+"-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
		+"-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
		+"-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
		+"-m cachesize : set cache memory size in MB (default 100)\n"
		+"-e epsilon : set tolerance of termination criterion (default 0.001)\n"
		+"-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
		+"-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
		+"-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
		+"-v n : n-fold cross validation mode\n"
		+"-q : quiet mode (no outputs)\n"
		+"\nMR-LIBSVM included:\n"
		+"-mr hadoop_uri mapreduce : mapreduce execution mode (NOTE: must provide URI of Hadoop cluster)\n"
		+"-cc n : n-subset cascade (default 16)\n"
		);
		System.exit(1);
	}

	private void do_cross_validation()
	{
		int i;
		int total_correct = 0;
		double total_error = 0;
		double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
		double[] target = new double[prob.l];

		svm.svm_cross_validation(prob,param,nr_fold,target);
		if(param.svm_type == svm_parameter.EPSILON_SVR ||
		   param.svm_type == svm_parameter.NU_SVR)
		{
			for(i=0;i<prob.l;i++)
			{
				double y = prob.y[i];
				double v = target[i];
				total_error += (v-y)*(v-y);
				sumv += v;
				sumy += y;
				sumvv += v*v;
				sumyy += y*y;
				sumvy += v*y;
			}
			System.out.print("Cross Validation Mean squared error = "+total_error/prob.l+"\n");
			System.out.print("Cross Validation Squared correlation coefficient = "+
				((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
				((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))+"\n"
				);
		}
		else
		{
			for(i=0;i<prob.l;i++)
				if(target[i] == prob.y[i])
					++total_correct;
			System.out.print("Cross Validation Accuracy = "+100.0*total_correct/prob.l+"%\n");
		}
	}
	
	private void run(String argv[]) throws IOException
	{
		parse_command_line(argv);
		read_problem();
		error_msg = svm.svm_check_parameter(prob,param);

		System.out.print(
                "------ LIBSVM normal execution mode ------\n"
               );
		if(error_msg != null)
		{
			System.err.print("ERROR: "+error_msg+"\n");
			System.exit(1);
		}

		if(cross_validation != 0)
		{
			do_cross_validation();
		}
		else
		{
			model = svm.svm_train(prob,param);
			svm.svm_save_model(model_file_name,model);
		}
	}
	
	/*
	 * MR-LIBSVM alternative run function.
	 * (Original LIBSVM will take the run() route.)
	 */
	
	private void run_mr(String argv[]) throws IOException, IllegalArgumentException, ClassNotFoundException, InterruptedException
	{
		System.out.print(
                "------ MR-LIBSVM cluster execution mode ------\n"
               +"Starting MapReduce jobs...\n"
               );
		parse_command_line(argv);
		generate_and_run_jobs();
	}
	
	/*
	 * MR-LIBSVM configure prepartition/cascade jobs and run them
	 */
	
	private void generate_and_run_jobs() throws IllegalArgumentException, IOException, ClassNotFoundException, InterruptedException
	{
		Configuration[] prepartition_confs = new Configuration[prepartition_job_count];
        for(int i=0; i<prepartition_job_count; i++){
                prepartition_confs[i] = new Configuration();
                prepartition_confs[i].set("yarn.resourcemanager.address", hadoop_address+":8032");
                prepartition_confs[i].set("yarn.resourcemanager.scheduler.address", hadoop_address+":8030");
                System.out.print("hdfs://"+hadoop_address+":9000\n");
                prepartition_confs[i].set("mapreduce.framework.name", "yarn");
                prepartition_confs[i].set("fs.default.name", "hdfs://"+hadoop_address+":9000");
                prepartition_confs[i].set("fs.hdfs.impl",org.apache.hadoop.hdfs.DistributedFileSystem.class.getName());
                prepartition_confs[i].set("fs.file.impl",org.apache.hadoop.fs.LocalFileSystem.class.getName());
        }
        
        prepartition_confs[0].setInt("SUBSET_COUNT",(int)subset_count);
        
        Configuration[] cascade_confs = new Configuration[cascade_job_count];
        
        // Mappers and Reducers will need to train SVM using user defined parameters,
        // so we're serializing the param object so we can add it to configuration.
        Gson paramGson = new Gson();
        String paramSerialization = paramGson.toJson(param);
        
        for(int i=0; i<cascade_job_count; i++) {
                cascade_confs[i] = new Configuration();
                cascade_confs[i].set("yarn.resourcemanager.address", hadoop_address+":8032");
                cascade_confs[i].set("yarn.resourcemanager.scheduler.address", hadoop_address+":8030");
                cascade_confs[i].set("mapreduce.framework.name", "yarn");
                cascade_confs[i].set("fs.default.name", "hdfs://"+hadoop_address+":9000");
                cascade_confs[i].set("fs.hdfs.impl",org.apache.hadoop.hdfs.DistributedFileSystem.class.getName());
                cascade_confs[i].set("fs.file.impl",org.apache.hadoop.fs.LocalFileSystem.class.getName());
                
                // Serializing param, and passing it via configuration.
                cascade_confs[i].set("PARAM", paramSerialization);
                
                // Mappers and Reducers use LIBSVM for training subroutine procedure
                DistributedCache.addFileToClassPath(new Path("libsvm.jar"), cascade_confs[i]);
        }
        
        
        prepartition_jobs = new Job[prepartition_job_count];
        
        prepartition_jobs[0] = new Job(prepartition_confs[0], "MR-LIBSVM: Partitioning training data, Phase 1");
        prepartition_jobs[0].setJarByClass(MRLibSVM.class);
        prepartition_jobs[0].setNumReduceTasks(0);
        prepartition_jobs[0].setMapperClass(PreStatCounterMapper.class);
        prepartition_jobs[0].setOutputKeyClass(NullWritable.class);
        prepartition_jobs[0].setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(prepartition_jobs[0], new Path(input_file_name));
        FileOutputFormat.setOutputPath(prepartition_jobs[0], new Path(output_hdfs_path+"/tmp"));
        
        System.out.print("\nPartitioning training data into "
                +subset_count
                +" subsets for cascade...\n");
        prepartition_jobs[0].waitForCompletion(false);
        prepartition_confs[1].setInt("SUBSET_COUNT",(int)subset_count);
        prepartition_confs[1].setInt("TOTAL_RECORD_COUNT",
                                (int)prepartition_jobs[0].getCounters().findCounter("trainingDataStats","TOTAL_RECORD_COUNT").getValue());
        prepartition_confs[1].setInt("CLASS_1_COUNT",
                                (int)prepartition_jobs[0].getCounters().findCounter("trainingDataStats","CLASS_1_COUNT").getValue());
        prepartition_confs[1].setInt("CLASS_2_COUNT",
                                (int)prepartition_jobs[0].getCounters().findCounter("trainingDataStats","CLASS_2_COUNT").getValue());

        prepartition_jobs[1] = new Job(prepartition_confs[1], "MR-LIBSVM: Partitioning training data, Phase 2");
        prepartition_jobs[1].setJarByClass(MRLibSVM.class);
        prepartition_jobs[1].setNumReduceTasks(subset_count);
        prepartition_jobs[1].setMapperClass(PrePartitionerMapper.class);
        prepartition_jobs[1].setReducerClass(SubsetDataOutputReducer.class);
        prepartition_jobs[1].setMapOutputKeyClass(IntWritable.class);
        prepartition_jobs[1].setMapOutputValueClass(Text.class);
        prepartition_jobs[1].setOutputKeyClass(NullWritable.class);
        prepartition_jobs[1].setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(prepartition_jobs[1], new Path(output_hdfs_path+"/tmp"));
        FileOutputFormat.setOutputPath(prepartition_jobs[1], new Path(output_hdfs_path+"/layer-input-subsets/layer-1"));
        prepartition_jobs[1].waitForCompletion(false);
        
        // Preparing cascade jobs...

        cascade_jobs = new Job[cascade_job_count];

        cascade_confs[cascade_job_count-1].set("USER_OUTPUT_PATH", output_hdfs_path);

        for(int i=0; i<cascade_job_count; i++) {
                cascade_jobs[i] = new Job(cascade_confs[i], "MR-LIBSVM: Layer " + (i+1));
                cascade_jobs[i].setJarByClass(MRLibSVM.class);
                cascade_jobs[i].setNumReduceTasks((int)(subset_count/Math.pow(2, i+1)));
                cascade_jobs[i].setMapperClass(SubSvmMapper.class);
                cascade_jobs[i].setMapOutputKeyClass(IntWritable.class);
                cascade_jobs[i].setMapOutputValueClass(Text.class);
                cascade_jobs[i].setOutputKeyClass(NullWritable.class);
                cascade_jobs[i].setOutputValueClass(Text.class);
                cascade_jobs[i].setInputFormatClass(TrainingSubsetInputFormat.class);
                FileInputFormat.addInputPath(cascade_jobs[i], new Path(output_hdfs_path+"/layer-input-subsets/layer-"+(i+1)));
                FileOutputFormat.setOutputPath(cascade_jobs[i], new Path(output_hdfs_path+"/layer-input-subsets/layer-"+(i+2)));

                if(i != cascade_job_count-1) {
                        cascade_jobs[i].setReducerClass(SubsetDataOutputReducer.class);
                } else {
                        cascade_jobs[i].setReducerClass(LastLayerSvmModelOutputReducer.class);
                }
        }

        System.out.print("\nStarting Cascade MapReduce SVM training framework...\n");
        
        for(int i=0; i<cascade_job_count; i++) {
            System.out.println("Beginning job for layer "+(i+1)+"...\n");
            if(i != cascade_job_count-1){
                    cascade_jobs[i].waitForCompletion(false);
                    System.out.println("Layer "+(i+1)+" has successfully completed!\n");
            } else {
                    int exit_stat = (cascade_jobs[i].waitForCompletion(false)) ? 0 : 1;
                    System.out.println("Layer "+(i+1)+" has succesfully completed!\n");
                    System.exit(exit_stat);
            }
        }
	}

	public static void main(String argv[]) throws IOException, IllegalArgumentException, ClassNotFoundException, InterruptedException
	{
		System.out.print(
		 "\n\n   __    __     ______     __         __     ______     ______     __   __   __    __   "
		+"\n  /\\ \"-./  \\   /\\  == \\   /\\ \\       /\\ \\   /\\  == \\   /\\  ___\\   /\\ \\ / /  /\\ \"-./  \\ " 
		+"\n  \\ \\ \\-./\\ \\  \\ \\  __<   \\ \\ \\____  \\ \\ \\  \\ \\  __<   \\ \\___  \\  \\ \\ \\\'/   \\ \\ \\-./\\ \\ "
		+"\n   \\ \\_\\ \\ \\_\\  \\ \\_\\ \\_\\  \\ \\_____\\  \\ \\_\\  \\ \\_____\\  \\/\\_____\\  \\ \\__|    \\ \\_\\ \\ \\_\\"
		+"\n    \\/_/  \\/_/   \\/_/ /_/   \\/_____/   \\/_/   \\/_____/   \\/_____/   \\/_/      \\/_/  \\/_/     \n\n"
		+"                                                          by Ting-Yun Tseng & Tzu-Li Tai\n\n\n"
		);   
		
		svm_train t = new svm_train();
		if(Arrays.asList(argv).contains("-mr")) {			
			t.run_mr(argv);
		} else {
			t.run(argv);
		}
	}

	private static double atof(String s)
	{
		double d = Double.valueOf(s).doubleValue();
		if (Double.isNaN(d) || Double.isInfinite(d))
		{
			System.err.print("NaN or Infinity in input\n");
			System.exit(1);
		}
		return(d);
	}

	private static int atoi(String s)
	{
		return Integer.parseInt(s);
	}

	private void parse_command_line(String argv[])
	{
		int i;
		svm_print_interface print_func = null;	// default printing to stdout

		param = new svm_parameter();
		// default values
		param.svm_type = svm_parameter.C_SVC;
		param.kernel_type = svm_parameter.RBF;
		param.degree = 3;
		param.gamma = 0;	// 1/num_features
		param.coef0 = 0;
		param.nu = 0.5;
		param.cache_size = 100;
		param.C = 1;
		param.eps = 1e-3;
		param.p = 0.1;
		param.shrinking = 1;
		param.probability = 0;
		param.nr_weight = 0;
		param.weight_label = new int[0];
		param.weight = new double[0];
		cross_validation = 0;
		is_hadoop_mode = false;
		subset_count = 16;

		// parse options
		for(i=0;i<argv.length;i++)
		{
			if(argv[i].charAt(0) != '-') break;
			if(++i>=argv.length)
				exit_with_help();
			switch(argv[i-1].charAt(1))
			{
				case 's':
					param.svm_type = atoi(argv[i]);
					break;
				case 't':
					param.kernel_type = atoi(argv[i]);
					break;
				case 'd':
					param.degree = atoi(argv[i]);
					break;
				case 'g':
					param.gamma = atof(argv[i]);
					break;
				case 'r':
					param.coef0 = atof(argv[i]);
					break;
				case 'n':
					param.nu = atof(argv[i]);
					break;
				case 'm':
					
					// MR-LIBSVM: -mr option
					if(argv[i-1].length() > 2) {
						hadoop_address = argv[i];
						is_hadoop_mode = true;
						
					// -m option
					} else {
						param.cache_size = atof(argv[i]);
					}
					break;
					
				case 'c':
					
					// MR-LIBSVM: -cc option
					if(argv[i-1].length() > 2) {
						if(!is_hadoop_mode) {
							System.err.print(
									 "Cascade: subset count specified when Hadoop mode is not enabled.\n"
									+"Please check if Hadoop mode is desired!\n");
							exit_with_help();
						} else {
							subset_count = atoi(argv[i]);
							prepartition_job_count = 2;
							cascade_job_count = (int)(Math.log(subset_count)/Math.log(2));
						}
					// -c option
					} else {
						param.C = atof(argv[i]);
					}
					break;
					
				case 'e':
					param.eps = atof(argv[i]);
					break;
				case 'p':
					param.p = atof(argv[i]);
					break;
				case 'h':
					param.shrinking = atoi(argv[i]);
					break;
				case 'b':
					param.probability = atoi(argv[i]);
					break;
				case 'q':
					print_func = svm_print_null;
					i--;
					break;
				case 'v':
					cross_validation = 1;
					nr_fold = atoi(argv[i]);
					if(nr_fold < 2)
					{
						System.err.print("n-fold cross validation: n must >= 2\n");
						exit_with_help();
					}
					break;
				case 'w':
					++param.nr_weight;
					{
						int[] old = param.weight_label;
						param.weight_label = new int[param.nr_weight];
						System.arraycopy(old,0,param.weight_label,0,param.nr_weight-1);
					}

					{
						double[] old = param.weight;
						param.weight = new double[param.nr_weight];
						System.arraycopy(old,0,param.weight,0,param.nr_weight-1);
					}

					param.weight_label[param.nr_weight-1] = atoi(argv[i-1].substring(2));
					param.weight[param.nr_weight-1] = atof(argv[i]);
					break;
				default:
					System.err.print("Unknown option: " + argv[i-1] + "\n");
					exit_with_help();
			}
		}

		svm.svm_set_print_string_function(print_func);

		// determine filenames

		if(i>=argv.length)
			exit_with_help();

		input_file_name = argv[i];
		output_hdfs_path = input_file_name+"_out";

		if(i<argv.length-1)
			model_file_name = argv[i+1];
		else
		{
			int p = argv[i].lastIndexOf('/');
			++p;	// whew...
			model_file_name = argv[i].substring(p)+".model";
		}
	}

	// read in a problem (in svmlight format)

	private void read_problem() throws IOException
	{
		BufferedReader fp = new BufferedReader(new FileReader(input_file_name));
		Vector<Double> vy = new Vector<Double>();
		Vector<svm_node[]> vx = new Vector<svm_node[]>();
		int max_index = 0;

		while(true)
		{
			String line = fp.readLine();
			if(line == null) break;

			StringTokenizer st = new StringTokenizer(line," \t\n\r\f:");

			vy.addElement(atof(st.nextToken()));
			int m = st.countTokens()/2;
			svm_node[] x = new svm_node[m];
			for(int j=0;j<m;j++)
			{
				x[j] = new svm_node();
				x[j].index = atoi(st.nextToken());
				x[j].value = atof(st.nextToken());
			}
			if(m>0) max_index = Math.max(max_index, x[m-1].index);
			vx.addElement(x);
		}

		prob = new svm_problem();
		prob.l = vy.size();
		prob.x = new svm_node[prob.l][];
		for(int i=0;i<prob.l;i++)
			prob.x[i] = vx.elementAt(i);
		prob.y = new double[prob.l];
		for(int i=0;i<prob.l;i++)
			prob.y[i] = vy.elementAt(i);

		if(param.gamma == 0 && max_index > 0)
			param.gamma = 1.0/max_index;

		if(param.kernel_type == svm_parameter.PRECOMPUTED)
			for(int i=0;i<prob.l;i++)
			{
				if (prob.x[i][0].index != 0)
				{
					System.err.print("Wrong kernel matrix: first column must be 0:sample_serial_number\n");
					System.exit(1);
				}
				if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
				{
					System.err.print("Wrong input format: sample_serial_number out of range\n");
					System.exit(1);
				}
			}

		fp.close();
	}
}