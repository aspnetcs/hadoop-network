package com.tencent.bi.cf.model.knn;

/**
 * Key,value pair
 * @author purlin
 *
 */
public class KVPair implements Comparable<KVPair>{
	/**
	 * Index
	 */
	private long id;
	/**
	 * Value
	 */
	private double val;
	
	public KVPair(){
		this.id = 0;
		this.val = 0.0;
	}
	
	public KVPair(long id, double val){
		this.id = id;
		this.val = val;
	}
	
	public long getId() {
		return id;
	}

	public void setId(long id) {
		this.id = id;
	}

	public double getVal() {
		return val;
	}

	public void setVal(double val) {
		this.val = val;
	}

	@Override
	public int compareTo(KVPair arg) {
		if(this.val == arg.getVal()) return 0;
		return (this.val - arg.getVal())<0.0 ? 1 : -1;
	}
}
