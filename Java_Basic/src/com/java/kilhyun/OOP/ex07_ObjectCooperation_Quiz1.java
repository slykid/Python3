package com.java.kilhyun.OOP;

public class ex07_ObjectCooperation_Quiz1 {

    /*
     * Quiz. 학생인 park 은 늦잠을 자서 학교에 지각을 했다. 때문에 택시를 타고, 10000월 지불하여 학교에 도착을 하였다.
     *       이 때 학생 park 이 학교에 도달하는 데 소요된 돈과 학교에 도착하는 데 까지 이용한 이동 수단의 수익 및 승객 수를 출력하는
     *       프로그램을 구현하시오.
     *       (단, Park 은 20000원을 가지고 있고, 택시는 번호가 없다고 가정한다.)
     */
    public static void main(String[] args)
    {
        Student_Prime studentPark = new Student_Prime("Park", 20000);

        Taxi taxi = new Taxi();

        studentPark.takeTaxi(taxi);
        taxi.takePassenger(10000);

        studentPark.showMoneyInfo();
        taxi.showTaxiInfo();


    }

}
