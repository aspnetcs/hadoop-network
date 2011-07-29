package com.tencent.bi.utils.matrix;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
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

import com.tencent.bi.utils.StringUtils;
import com.tencent.bi.utils.hadoop.FileOperators;
import com.tencent.bi.utils.serialization.MatrixRowWritable;

public class SparseDenseMul implements MatrixOperator{
		
	@Override
	public void perform(String inputA, String inputB, String outputPath)
			throws Exception {
		//First MR
		Configuration conf = FileOperators.getConfiguration();
		conf.set("path.A", new Path(inputA).getName());
		Job job = new Job(conf);
		job.setJarByClass(SparseDenseMul.class);
		job.setJobName("SparseDenseMul-phase1");	
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
		FileOutputFormat.setOutputPath(job, new Path(outputPath));
		job.waitForCompletion(true);
		FileSystem fs = FileSystem.get(conf);
		fs.delete(new Path(conf.get("hadoop.tmp.path")+"SparseDenseMul/"),true);
	}

///////////////////////////////////////////////////////////////////////////////////////////////
	
	public static class RowMapper extends Mapper<LongWritable, Text, LongWritable, MatrixRowWritable>{
		
		private LongWritable outKey = new LongWritable();
		
		private MatrixRowWritable outValue = new MatrixRowWritable();
		
		private String inputA = "";
		
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			String path = ((FileSplit) context.getInputSplit()).getPath().toString();
			String[] items = value.toString().split(",");
			outKey.set(Long.parseLong(items[0]));
			if(path.indexOf(inputA)>-1){	//Sparse
				long id = Long.parseLong(items[1]);
				double val = Double.parseDouble(items[2]);
				outValue.set(id, val);
			} else { //Dense
				double[] vec = new double[items.length-1];
				for(int i=0;i<vec.length;i++){
					vec[i] = Double.parseDouble(items[i+1]);
				}
				outValue.set(vec);
			}
			context.write(outKey, outValue);
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			Configuration conf = context.getConfiguration();
			inputA = conf.get("input.A", "");
		}
	}
	
	public static class RowReducer extends Reducer<LongWritable, MatrixRowWritable, LongWritable, MatrixRowWritable> {
		
		private LongWritable outKey = new LongWritable();
		
		private MatrixRowWritable outValue = new MatrixRowWritable();
		
		@Override
		public void reduce(LongWritable key, Iterable<MatrixRowWritable> values, Context context) throws IOException, InterruptedException {
			//Dense vector
			double[] vec = null;
			//Sparse values
			Map<Long, Double> vList = new HashMap<Long, Double>();
			for(MatrixRowWritable value: values){
				if(value.isSparse()){
					vList.put(value.getFirstID(), value.getFirstVal());
				} else {
					vec = value.getVector();
				}
			}
			//Output
			double[] res = new double[vec.length];
			for(Long k : vList.keySet()){
				for(int i=0;i<vec.length;i++)
					res[i] = vec[i]*vList.get(k);
				outKey.set(k);
				outValue.set(res);
				context.write(outKey, outValue);
			}
		}
	}

///////////////////////////////////////////////////////////////////////////////////////////////
	
	public static class ColReducer extends Reducer<LongWritable, MatrixRowWritable, LongWritable, Text> {
				
		private Text outValue = new Text();
		
		@Override
		public void reduce(LongWritable key, Iterable<MatrixRowWritable> values, Context context) throws IOException, InterruptedException {
			double[] vec = null;
			for(MatrixRowWritable value : values){ //Combine
				if(vec==null)
					vec = new double[(int) value.getLen()];
				double[] cv = value.getVector();
				for(int i=0;i<cv.length;i++)
					vec[i] = vec[i] + cv[i];
			}
			//Output
			outValue.set(StringUtils.array2String(vec));
			context.write(key, outValue);
		}
	}
}
