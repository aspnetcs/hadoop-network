package com.tencent.bi.cf.model.knn;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;

import com.tencent.bi.utils.hadoop.FileOperators;
import com.tencent.bi.utils.serialization.KeyPairWritable;
import com.tencent.bi.utils.serialization.MatrixRowWritable;

/**
 * Build inverse index
 * @author tigerzhong
 *
 */
public class InverseIndexBuilder {
	/**
	 * Builder
	 * @param inputPath, input path of data
	 * @param K, number of neighbors 
	 * @param criterion, name of similarity criterion
	 * @param lambda, reguralization parameter
	 * @param userBase, user-base of item-base
	 * @param m, number of users
	 * @param n, number of items
	 * @throws IOException
	 * @throws InterruptedException
	 * @throws ClassNotFoundException
	 */
	public static void build(String inputPath, int K, String criterion, double lambda, boolean userBase, int m, int n) throws IOException, InterruptedException, ClassNotFoundException{
		Configuration conf = FileOperators.getConfiguration(); 
		conf.setInt("model.k", K);
		conf.setInt("model.m", m);
		conf.setInt("model.n", n);
		conf.set("model.criterion", criterion);
		conf.setFloat("model.lambda", (float) lambda);
		conf.setBoolean("model.userbase", userBase);
		Job job = new Job(conf);
		job.setJobName("KNN-InverseIndex");
		job.setJarByClass(InverseIndexBuilder.class);
		job.setMapOutputKeyClass(LongWritable.class);
		job.setMapOutputValueClass(KeyPairWritable.class);
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(MatrixRowWritable.class);
		job.setMapperClass(InverseIndexMapper.class);
		job.setReducerClass(InverseIndexReducer.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		FileInputFormat.setInputPaths(job, new Path(inputPath));
		FileOutputFormat.setOutputPath(job, new Path(conf.get("hadoop.cache.path")+"InverseIndex/"));
		job.waitForCompletion(true);
	}
	
	/**
	 * Mapper for building inverse index
	 * @author tigerzhong
	 *
	 */
	public static class InverseIndexMapper extends Mapper<LongWritable, Text, LongWritable, KeyPairWritable> {
		/**
		 * Key for output
		 */
		private LongWritable outKey = new LongWritable();
		/**
		 * Value for output
		 */
		private KeyPairWritable outValue = new KeyPairWritable();
		/**
		 * User-base or Item-base?
		 */
		private boolean userBase = true;

		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			String items[] = value.toString().split(",",-1);
			if(userBase){
				outKey.set(Long.parseLong(items[1]));
				outValue.set(Long.parseLong(items[0]), Double.parseDouble(items[2]));
			}
			else {
				outKey.set(Long.parseLong(items[0]));
				outValue.set(Long.parseLong(items[1]), Double.parseDouble(items[2]));			
			}
			context.write(outKey, outValue);
		}
		
		@Override
		public void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			userBase = context.getConfiguration().getBoolean("model.userbase", true);
		}
	}
	
	/**
	 * Reducer for building inverse index
	 * @author tigerzhong
	 *
	 */
	public static class InverseIndexReducer extends Reducer<LongWritable, KeyPairWritable, LongWritable, MatrixRowWritable> {
		
		/**
		 * Output value
		 */
		private MatrixRowWritable outValue = new MatrixRowWritable();

		private List<Double> valueList = new ArrayList<Double>();
		
		private List<Long> idList = new ArrayList<Long>();
		
		private int num = 0;
		@Override
		public void reduce(LongWritable key, Iterable<KeyPairWritable> values, Context context)
			throws IOException, InterruptedException {
			valueList.clear();
			idList.clear();
			for(KeyPairWritable value : values){
				valueList.add(value.getValue());
				idList.add(value.getKey());
			}
			outValue.set(idList, valueList, num);
			context.write(key, outValue);
		}
		
		@Override
		public void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			boolean userBase = context.getConfiguration().getBoolean("model.userbase", true);
			if(userBase) num = context.getConfiguration().getInt("model.m", 0);
			else num = context.getConfiguration().getInt("model.n", 0);
		}
	}
}
