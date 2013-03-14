package com.tencent.bi.cf.model.knn;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import com.tencent.bi.cf.model.knn.criterion.Criterion;
import com.tencent.bi.utils.hadoop.FileOperators;
import com.tencent.bi.utils.serialization.DoublePairWritable;
import com.tencent.bi.utils.serialization.KeyPairWritable;
import com.tencent.bi.utils.serialization.LongPairWritable;
import com.tencent.bi.utils.serialization.MatrixRowWritable;

/**
 * Neighborhood builder
 * @author tigerzhong
 *
 */
public class NgBuilder {

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
	public static void build(int K, String criterion, double lambda, boolean userBase, int m, int n) throws IOException, InterruptedException, ClassNotFoundException{
		Configuration conf = FileOperators.getConfiguration(); 
		conf.setInt("model.k", K);
		conf.setInt("model.m", m);
		conf.setInt("model.n", n);
		conf.set("model.criterion", criterion);
		conf.setFloat("model.lambda", (float) lambda);
		conf.setBoolean("model.userbase", userBase);
		//Similarity
		Job job = new Job(conf);
		job.setJobName("KNN-Similarity");
		job.setJarByClass(NgBuilder.class);
		job.setMapOutputKeyClass(LongPairWritable.class);
		job.setMapOutputValueClass(DoublePairWritable.class);
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(KeyPairWritable.class);
		job.setMapperClass(SimilarityMapper.class);
		job.setReducerClass(SimilarityReducer.class);
		job.setInputFormatClass(SequenceFileInputFormat.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		FileInputFormat.setInputPaths(job, new Path(conf.get("hadoop.cache.path")+"InverseIndex/"));
		FileOutputFormat.setOutputPath(job, new Path(conf.get("hadoop.cache.path")+"SimList/"));
		job.waitForCompletion(true);
		//Sort
		conf.set("mapred.textoutputformat.separator", ",");
		job = new Job(conf);
		job.setJobName("KNN-Ng");
		job.setJarByClass(NgBuilder.class);
		job.setMapOutputKeyClass(LongWritable.class);
		job.setMapOutputValueClass(KeyPairWritable.class);
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(Text.class);
		job.setMapperClass(Mapper.class);
		job.setReducerClass(NgReducer.class);
		job.setInputFormatClass(SequenceFileInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		FileInputFormat.setInputPaths(job, new Path(conf.get("hadoop.cache.path")+"SimList/"));
		FileOutputFormat.setOutputPath(job, new Path(conf.get("hadoop.cache.path")+"NgList/"));
		job.waitForCompletion(true);
		
	}
	
	/**
	 * Mapper for buiding the similarity between users or items
	 * @author tigerzhong
	 *
	 */
	public static class SimilarityMapper extends Mapper<LongWritable, MatrixRowWritable, LongPairWritable, DoublePairWritable> {
		/**
		 * Key for output
		 */
		private LongPairWritable outKey = new LongPairWritable();
		
		/**
		 * Value for output
		 */
		private DoublePairWritable outValue = new DoublePairWritable();

		@Override
		public void map(LongWritable key, MatrixRowWritable value, Context context)
				throws IOException, InterruptedException {
			long[] idList = value.getIDs();
			double[] valueList = value.getVector();
			for(int i=0;i<idList.length;i++){
				for(int j=0;j<idList.length;j++){
					if(i==j) continue;
					outKey.set(idList[i], idList[j]);
					outValue.set(valueList[i], valueList[j]);
					context.write(outKey, outValue);
				}
			}
		}
	}
	
	/**
	 * Reducer for buiding the similarity between users or items
	 * @author tigerzhong
	 *
	 */
	public static class SimilarityReducer extends Reducer<LongPairWritable, DoublePairWritable, LongWritable, KeyPairWritable> {
		
		/**
		 * Value for output
		 */
		private KeyPairWritable outValue = new KeyPairWritable();
		
		private LongWritable outKey = new LongWritable();
		
		protected float lambda;
		
		protected Criterion c;

		protected int num;
		
		protected List<Double> vecA = new ArrayList<Double>();
		
		protected List<Double> vecB = new ArrayList<Double>();
		
		@Override
		public void setup(Context context) {
			try {
				Configuration conf = context.getConfiguration();	
				lambda = conf.getFloat("model.lambda", 100.0f);
				c = (Criterion) Class.forName(conf.get("model.criterion")).newInstance();
				boolean userBase = context.getConfiguration().getBoolean("model.userbase", true);
				if(userBase) num = context.getConfiguration().getInt("model.m", 0);
				else num = context.getConfiguration().getInt("model.n", 0);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		@Override
		public void reduce(LongPairWritable key, Iterable<DoublePairWritable> values, Context context)
				throws IOException, InterruptedException {
			vecA.clear(); vecB.clear();
			for(DoublePairWritable value: values){
				vecA.add(value.getFirst());
				vecB.add(value.getSecond());
			}			
			double sim = c.getValue(vecA, vecB, lambda);
			outValue.set(key.getSecond(), sim);
			outKey.set(key.getFirst());
			context.write(outKey, outValue);
		}
	}
	
	/**
	 * Reducer for finding the neighbors of users or items
	 * @author tigerzhong
	 *
	 */
	public static class NgReducer extends Reducer<LongWritable, KeyPairWritable, LongWritable, Text> {
		
		/**
		 * Value for output
		 */
		private Text outValue = new Text();
		
		private List<KVPair> vec = new ArrayList<KVPair>();
		
		private KVPair kv = new KVPair();
		
		protected int K;
		
		@Override
		public void setup(Context context) {
			Configuration conf = context.getConfiguration();	
			K = conf.getInt("model.K", 121);
		}

		@Override
		public void reduce(LongWritable key, Iterable<KeyPairWritable> values, Context context)
				throws IOException, InterruptedException {
			for(KeyPairWritable value : values){
				kv.setId(value.getKey());
				kv.setVal(value.getValue());
				vec.add(kv);
			}
			Collections.sort(vec);
			StringBuilder outLine = new StringBuilder("");
			for(int i=0;i<vec.size() && i<K;i++){
				if(i!=0) outLine.append(",");
				kv = vec.get(i);
				outLine.append(kv.getId()+",");
				outLine.append(kv.getVal());
			}
			outValue.set(outLine.toString());
			context.write(key, outValue);
		}
	}
}
