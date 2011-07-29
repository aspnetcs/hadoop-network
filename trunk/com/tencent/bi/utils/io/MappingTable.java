package com.tencent.bi.utils.io;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import com.tencent.bi.cf.model.common.MatrixInitialization;
import com.tencent.bi.utils.hadoop.FileOperators;

/**
 * Mapping QQnum and adID into a fix range
 * @author tigerzhong
 *
 */
public class MappingTable {
	/**
	 * Column ID
	 */
	private int colID = 0;
	/**
	 * Input path name
	 */
	private String inputName = "";
	/**
	 * Output path name
	 */
	private String outputName = "";
	/**
	 * Initialization
	 * @param cid
	 * @param inName
	 * @param outName
	 * @throws IOException
	 */
	public void init(int cid, String inName, String outName) throws IOException{
		colID = cid;
		inputName = inName;
		outputName = outName;
	}
	/**
	 * Perform mapping
	 * @throws IOException
	 * @throws InterruptedException
	 * @throws ClassNotFoundException
	 */
	public void perform() throws IOException, InterruptedException, ClassNotFoundException{
		Configuration conf = FileOperators.getConfiguration();
		conf.setInt("matrix.colID", this.colID);
		Job job = new Job(conf);
		job.setJarByClass(MatrixInitialization.class);
		job.setJobName("Building-Mapping");
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		job.setMapperClass(BuildingMapper.class);
		job.setReducerClass(BuildingReducer.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(inputName));
		FileOutputFormat.setOutputPath(job, new Path(conf.get("hadoop.tmp.path")));
		job.waitForCompletion(true);
		postProcess(conf.get("hadoop.tmp.path"), outputName, job.getConfiguration());
		//FileSystem fs = FileSystem.get(job.getConfiguration());
		//fs.delete(new Path(Constant.TEMP_PATH),true);
	}
	
	protected void postProcess(String inName, String outName, Configuration conf) throws IOException{
		FileSystem fs = FileSystem.get(conf); 
		FileStatus fsta[] = fs.globStatus(new Path(inName+"*"));
		FSDataOutputStream ofs = fs.create(new Path(outName));
		int cnt = 0;
		for (FileStatus it : fsta) {
			Path singlePath = it.getPath();
			if(it.isDir()) continue;
			BufferedReader in = new BufferedReader(new InputStreamReader(fs.open(singlePath)));
			int cpt = 0;
			while(in.ready()){
				String[] items = in.readLine().split("\t");
				long k = Long.parseLong(items[0]);
				int mv = Integer.parseInt(items[1]);
				String out = k+","+(cnt+mv)+"\n";
				ofs.writeChars(out);
				cpt++;
			}
			in.close();
			cnt += cpt;
			System.out.println(cnt);
		}
		ofs.close();
	}
	
	public static class BuildingMapper extends Mapper<LongWritable, Text, Text, Text> {
		
		private int colID = 0;
		/**
		 * Key for output
		 */
		private static Text outKey = new Text();
		/**
		 * Output value
		 */
		private Text outText = new Text();
		
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			String[] items = value.toString().split(",");
			long id = Long.parseLong(items[colID]);
			long outkey = id / 100;
			outKey.set(outkey+"");
			outText.set(id+"");
			context.write(outKey, outText);
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			colID = context.getConfiguration().getInt("matrix.colID", 0);
		}
	}
	
	public static class BuildingReducer extends Reducer<Text, Text, Text, Text> {
		/**
		 * Key for output
		 */
		private static Text outKey = new Text();
		/**
		 * Output value
		 */
		private static Text outText = new Text();
		
		@Override
		public void reduce(Text key, Iterable<Text> values,
				Context context) throws IOException, InterruptedException {
			Iterator<Text> it = values.iterator();
			int i=0;
			String pre = "";
			while(it.hasNext()){
				String current = it.next().toString();
				if(current.equals(pre)) continue;
				pre = current;
				outKey.set(current);
				outText.set(i+"");
				i++;
				context.write(outKey,outText);
			}
		}
	}
}
