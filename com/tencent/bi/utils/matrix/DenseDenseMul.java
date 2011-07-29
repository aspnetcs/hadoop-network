package com.tencent.bi.utils.matrix;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import com.tencent.bi.utils.hadoop.FileOperators;
import com.tencent.bi.utils.serialization.MatrixRowWritable;
import com.tencent.bi.utils.serialization.PairRowWritable;
import com.tencent.bi.utils.serialization.LongPairWritable;

public class DenseDenseMul implements MatrixOperator {

	@Override
	public void perform(String inputA, String inputB, String output)
			throws Exception {
		//First MR
		Configuration conf = FileOperators.getConfiguration();
		conf.set("path.A", new Path(inputA).getName());
		Job job = new Job(conf);
		job.setJarByClass(SparseDenseMul.class);
		job.setJobName("DenseDenseMul-phase1");	
		job.setMapOutputKeyClass(LongWritable.class);
		job.setMapOutputValueClass(MatrixRowWritable.class);
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(MatrixRowWritable.class);
		job.setMapperClass(RowMapper.class);
		job.setReducerClass(RowReducer.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(inputA));
		FileInputFormat.addInputPath(job, new Path(inputB));
		FileOutputFormat.setOutputPath(job, new Path(conf.get("hadoop.tmp.path")+"SparseDenseMul/"));
		job.waitForCompletion(true);
		//Second MR
		conf.set("mapred.textoutputformat.separator", ",");
		job = new Job(conf);
		job.setJarByClass(SparseDenseMul.class);
		job.setJobName("SparseDenseMul-phase2");	
		job.setMapOutputKeyClass(LongWritable.class);
		job.setMapOutputValueClass(MatrixRowWritable.class);
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(MatrixRowWritable.class);
		job.setMapperClass(Mapper.class);
		job.setReducerClass(ColReducer.class);
		job.setInputFormatClass(SequenceFileInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.tmp.path")+"SparseDenseMul/"));
		FileOutputFormat.setOutputPath(job, new Path(output));
		job.waitForCompletion(true);
	}
	
///////////////////////////////////////////////////////////////////////////////////////////////
	
	public static class RowMapper extends Mapper<LongWritable, Text, LongWritable, PairRowWritable>{
		
		private LongWritable outKey = new LongWritable();
		
		private PairRowWritable outValue = new PairRowWritable();
		
		private String inputA = "";
		
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			String path = ((FileSplit) context.getInputSplit()).getPath().toString();
			String[] items = value.toString().split(",");
			//Get the whole row
			if(path.indexOf(inputA)>-1){	//First
				double[] tag = new double[1]; tag[0] = -1;
				outValue.setFectors(tag);
				for(int i=0;i<items.length-1;i++){
					outKey.set(i);
					outValue.setRating(Long.parseLong(items[0]), Double.parseDouble(items[i+1]));
					context.write(outKey, outValue);
				}				
			} else { 						//Second
				outKey.set(Long.parseLong(items[0]));
				double[] tag = new double[1]; tag[0] = 1;
				outValue.setFectors(tag);
				for(int i=0;i<items.length-1;i++){
					outValue.setRating(i, Double.parseDouble(items[i+1]));
					context.write(outKey, outValue);
				}
			}
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			Configuration conf = context.getConfiguration();
			inputA = conf.get("path.A", "");
		}
	}
	
	public static class RowReducer extends Reducer<LongWritable, PairRowWritable, LongPairWritable, DoubleWritable> {
		
		private LongPairWritable outKey = new LongPairWritable();
		
		private DoubleWritable outValue = new DoubleWritable();
		
		private Map<Long, Double> rowA = new HashMap<Long, Double>();
		
		private Map<Long, Double> rowB = new HashMap<Long, Double>();
		
		@Override
		public void reduce(LongWritable key, Iterable<PairRowWritable> values, Context context) throws IOException, InterruptedException {
			for(PairRowWritable value: values){
				if(value.getFactorArray()[0]>0)
					rowB.put(value.getRowIDs()[0], value.getRatings()[0]);
				else rowA.put(value.getRowIDs()[0], value.getRatings()[0]);
			}
			//Output
			for(Long k1: rowA.keySet()){
				for(Long k2: rowB.keySet()){
					outKey.set(k1, k2);
					outValue.set(rowA.get(k1)*rowB.get(k2));
					context.write(outKey, outValue);
				}
			}
		}
	}

///////////////////////////////////////////////////////////////////////////////////////////////
	
	public static class ColReducer extends Reducer<LongPairWritable, DoubleWritable, LongPairWritable, DoubleWritable> {
		
		private DoubleWritable outValue = new DoubleWritable();
		
		@Override
		public void reduce(LongPairWritable key, Iterable<DoubleWritable> values, Context context) throws IOException, InterruptedException {
			double sum = 0.0;
			for(DoubleWritable value : values) //Combine
				sum += value.get();
			outValue.set(sum);
			context.write(key, outValue);
		}
	}
}
