package com.java.kilhyun.OOP;

public class Subway {

    int lineNo; 		// 지하철노선
    int passengerCnt;	// 승객 수
    int income;			// 수입

    public Subway(int lineNo)
    {
        this.lineNo = lineNo;
    }

    public void takePassenger(int charge)
    {
        this.income += charge;
        passengerCnt++;
    }

    // 현재 수입 및 승객 수 확인
    public void showSubwayInfo()
    {
        System.out.println("지하철 " + lineNo + " 호선의 승객 수는 " + passengerCnt + " 명이고 수입은 " + income + " 원 입니다.");
    }

}
