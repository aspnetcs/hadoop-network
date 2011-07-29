package com.tencent.bi.driver;

import com.tencent.bi.cf.model.naive.RelationalPredictionEnhanced;
import com.tencent.bi.utils.hadoop.FileOperators;

//import java.io.IOException;

//import com.tencent.mf.model.naive.Bias;
//import com.tencent.bi.utils.io.MatrixIO;
//import com.tencent.bi.cf.model.common.MatrixInitialization;



public class TestDriver {

	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws Exception {
		
		/*// Matrix init
		int nr = Integer.parseInt(args[0]);
		int nc = Integer.parseInt(args[1]);
		String path = args[2];
		MatrixInitialization obj = new MatrixInitialization();
		obj.init(nr, nc, path, true);
		obj.perform();
		//Building mapping table
		int cid = Integer.parseInt(args[0]);
		String inName = args[1];
		String outName = args[2];
		MappingTable obj = new MappingTable();
		obj.init(cid, inName, outName);
		obj.perform();		
		Bias.getBias(args[0], Constant.TEMP_PATH+"BiasTmp/");*/
		//MatrixIO.readMatrix(args[0]);
		FileOperators.confName = args[3];
		RelationalPredictionEnhanced model = new RelationalPredictionEnhanced();
		model.initModel(args[0], args[1], Integer.parseInt(args[2]));
		model.buildModel();
	}

}
