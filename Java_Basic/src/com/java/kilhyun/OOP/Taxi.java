package com.java.kilhyun.OOP;

public class Taxi {

    int passengerCnt;	// 승객 수
    int income;			// 수입

    public Taxi() {}

    // 승객 탑승
    public void takePassenger(int charge)
    {
        this.income += charge;
        passengerCnt++;
    }

    // 현재 수입 및 승객 수 확인
    public void showTaxiInfo()
    {
        System.out.println("택시의 승객 수는 " + passengerCnt + " 명이고 수입은 " + income + " 원 입니다.");
    }

}
