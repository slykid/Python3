package com.java.kilhyun.OOP;

public class ex07_ObjectCooperation {
    /*
     * 객체간 협력
     * - 객체지향 프로그래밍이란, 객체를 정의하고 객체간의 협력을 구현한 프로그램을 구현하는 것이다.
     * - 객체 간의 협업을 구현하려면, 반드시 객체가 먼저 생성되어있어야 한다.
     *
     * 구현 예시
     * - 학생이 학교에 가는 방법을 정의하며, 이동수단은 지하철과 버스만 이용하는 것으로 정의한다.
     * - 학생, 버스, 지하철에 대한 클래스 생성
     * - 학생이 학교까지 가는 데 필요한 행위를 메소드로 구현
     *   - 이동 수단에 대한 비용 부담, 버스 및 지하철의 수입과 승객 수 출력 등
     *
     * - 구현에 필요한 모든 클래스는 cooperate 라는 패키지 이하의 클래스로 생성한다.
     *
     */
    public static void main(String[] args)
    {
        // 학생
        Student_Prime studentKim = new Student_Prime("Kim", 10000);
        Student_Prime studentLee = new Student_Prime("Lee", 15000);

        // 버스
        Bus bus60 = new Bus(60);
        Bus bus9501 = new Bus(9501);

        // 지하철
        Subway subwayLine5 = new Subway(5);
        Subway subwayLine2 = new Subway(2);


        // 교통수단 이용
        studentKim.takeBus(bus60);
        studentLee.takeSubway(subwayLine5);


        // 결과 출력
        System.out.println("============학생 잔액 확인=============");
        studentKim.showMoneyInfo();
        studentLee.showMoneyInfo();
        System.out.println();

        System.out.println("============교통수단 정보 확인=============");
        bus60.showBusInfo();
        bus9501.showBusInfo();
        System.out.println();

        subwayLine2.showSubwayInfo();
        subwayLine5.showSubwayInfo();
    }

}
