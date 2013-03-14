package com.tencent.bi.graph.model.search;

import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;

import com.tencent.bi.utils.hadoop.DataOperators;
import com.tencent.bi.utils.hadoop.FileOperators;
import com.tencent.bi.utils.io.MatrixIO;
import com.tencent.bi.utils.serialization.MatrixRowWritable;

/**
 * Get the number of hops between two nodes
 * @author tigerzhong
 *
 */
public class Reachable extends BFS{
	/**
	 * The distance between two nodes
	 */
	private double dist = 0.0;
	/**
	 * The ID of the target node
	 */
	private long endID;

	
	/**
	 * Get the distant
	 * @param idA
	 * @param idB
	 * @return distance between A and B
	 * @throws Exception 
	 */
	public double run(long idA, long idB) throws Exception{
		MatrixRowWritable value = new MatrixRowWritable();
		value.set(0.0);
		Configuration conf = FileOperators.getConfiguration();
		MatrixIO.writeMatrixRow(conf.get("hadoop.cache.path")+"DistBFS/0/", new LongWritable(idA), value);
		this.endID = idB;
		performSearch();
		return this.dist;
	}
	
	@Override
	protected void visit(long nodeID, double dist) throws Exception{
		if(nodeID==endID){
			this.dist = dist;
			//Write finished flag to HDFS
			List<String> out = new ArrayList<String>();
			out.add("F");
			Configuration conf = FileOperators.getConfiguration();
			DataOperators.saveTextToHDFS(new Configuration(), conf.get("hadoop.tmp.path")+"Reachable.finished", out, true);
		}
	}

	@Override
	protected boolean isFinished() throws Exception {
		FileSystem fs = FileSystem.get(new Configuration());
		Configuration conf = FileOperators.getConfiguration();
		return fs.exists(new Path(conf.get("hadoop.tmp.path")+"Reachable.finished"));
	}

}
