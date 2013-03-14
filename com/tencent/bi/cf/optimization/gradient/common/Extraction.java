package com.tencent.bi.cf.optimization.gradient.common;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;

import cern.colt.matrix.impl.DenseDoubleMatrix2D;

import com.tencent.bi.utils.serialization.LongPairWritable;
import com.tencent.bi.utils.serialization.MatrixRowWritable;

public class Extraction {

	/**
	 * Extract V from file
	 * @param V, latent matrix for items
	 * @param conf, configuration
	 * @param pathName, data path name
	 * @throws IOException
	 */
	public static void extract(DenseDoubleMatrix2D V, Configuration conf, String pathName, double learningRate) throws IOException{
		//Update V
		FileSystem fs = FileSystem.get(conf); 
		FileStatus fsta[] = fs.globStatus(new Path(pathName+"*"));
		LongPairWritable key = new LongPairWritable();
		MatrixRowWritable value = new MatrixRowWritable();
		for (FileStatus it : fsta) {
			Path singlePath = it.getPath();
			if(it.isDir()) continue;
			SequenceFile.Reader rd = new SequenceFile.Reader(fs, singlePath, new Configuration());
			while(rd.next(key, value)){				//Processing line by line
				double[] vec = value.viewVector();
				int k = (int)key.getFirst();
				if(key.getSecond()==0){				//V
					for(int i=0;i<vec.length;i++){
						V.set(k,i, V.get(k, i) - learningRate*vec[i]);
					}
				}
			}
			rd.close();
		}
	}
	
}
