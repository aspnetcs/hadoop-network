package com.tencent.bi.utils;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import com.tencent.bi.utils.hadoop.FileOperators;

import cern.colt.matrix.DoubleMatrix2D;

/**
 * A utility class providing a few static methods for string manipulation.
 * @author tigerzhong
 */
public class StringUtils {
    /**
     * Use a MR job to "change" the type of key of input data.
     * @weixue Is is really necessary?
     * @param inputPath
     * @param outputPath
     * @throws Exception
     */
	public static void modifyKey(String inputPath, String outputPath) throws Exception{
		Configuration conf = FileOperators.getConfiguration();
		conf.set("mapred.textoutputformat.separator", ",");
		Job job = new Job(conf);
		job.setJarByClass(StringUtils.class);
		job.setJobName("Modify-Key");		
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		job.setMapperClass(ModifyMapper.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(inputPath));
		FileOutputFormat.setOutputPath(job, new Path(outputPath));
		job.waitForCompletion(true);
	}
	
	/**
	 * Mapper used to change the key from {@link LongWritable} to {@link Text}.
	 * @author tigerzhong
	 */
	public static class ModifyMapper extends Mapper<LongWritable, Text, Text, Text>{
		/**
		 * Output Key
		 */
		private Text outKey = new Text();
		/**
		 * Output value
		 */
		private Text outText = new Text();
		
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			String[] items = value.toString().split(",",2);
			// @weixue Why length()-2? Give an example of input lines please.
			outKey.set(items[0].substring(0, items[0].length()-2));
			outText.set(items[1]);
			context.write(outKey, outText);
		}
	}
	
	/**
	 * @param array
	 * @return the string encoding of the input double array.
	 */
	public static String array2String(double[] array) {
		StringBuilder res = new StringBuilder("");
		for (int i = 0; i < array.length; i++) {
			if (i != 0)
				res.append(",");
			res.append(array[i]);
		}
		return res.toString();
	}

	/**
	 * @param M
	 * @return the string encoding of the input 2D double matrix.
	 */
	public static String matrix2String(DoubleMatrix2D M) {
		StringBuilder res = new StringBuilder("");
		for (int i = 0; i < M.rows(); i++)
			for (int j = 0; j < M.columns(); j++) {
				if (i != 0 || j != 0)
					res.append(",");
				res.append(M.getQuick(i, j));
			}
		return res.toString();
	}

}
