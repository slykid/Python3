package com.java.kilhyun.OOP;

public class MyDate {

    private int day;
    private int month;
    private int year;

    // getter, setter 선언
    public int getDay() {
        return day;
    }

    public int getMonth() {
        return month;
    }

    public int getYear() {
        return year;
    }


    public void setDay(int day) {
        this.day = day;
    }

    public void setMonth(int month) {
        this.month = month;
    }

    public void setYear(int year) {
        this.year = year;
    }

    // 날짜 출력 함수
    public void showDate()
    {
        System.out.println(year +"년 "+ month +"월 "+ day +"일 입니다.");
    }

}
