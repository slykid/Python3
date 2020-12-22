package com.java.kilhyun.OOP;

public class Student_Prime {

    String studentName;  // 학생 이름
    int grade;			 // 학년
    int money;			 // 소유 금액

    public Student_Prime(String studentName, int money)
    {
        this.studentName = studentName;
        this.money = money;
    }

    // 객체간 협업 : 버스 탑승
    public void takeBus(Bus bus)
    {
        bus.takePassenger(1000);
        this.money -= 1000;
    }

    // 객체간 협업 : 지하철 탑승
    public void takeSubway(Subway subway)
    {
        subway.takePassenger(1200);
        this.money -= 1200;
    }

    // 객체간 협업 : 택시 (Quiz 부분)
    public void takeTaxi(Taxi taxi)
    {
        taxi.takePassenger(10000);
        this.money -= 10000;
    }

    // 잔액 확인
    public void showMoneyInfo()
    {
        System.out.println(studentName + " 님의 잔액은 " + money + " 원 입니다.");
    }
}
