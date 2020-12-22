package com.java.kilhyun.OOP;

public class etc_CalculateSquare {

    public static void main(String[] args)
    {
        etc_AreaCalculator areaCalc = new etc_AreaCalculator();

        System.out.println("길이 3.5 인 정사각형의 넓이 : " + areaCalc.calculateRectangleArea(3.5));
        System.out.println("가로 길이 3.5, 세로 길이 4 인 직사각형의 넓이 : " + areaCalc.calculateRectangleArea(3.5, 4));
    }

}
