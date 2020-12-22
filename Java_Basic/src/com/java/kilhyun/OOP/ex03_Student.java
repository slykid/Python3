package com.java.kilhyun.OOP;

public class ex03_Student {
    /*
     * 메소드
     * - 객체의 기능을 구현하기 위해 클래스 내부에서 구현되는 함수
     * - 메소드의 구현 = 객체의 기능 구현
     * - 명명법은 사용하는 쪽(클라이언트 코드)에 맞게 명명함
     *   ex. getStudentName()
     */

    // 학생(학번, 이름, 주소)
    public int studentId = 0;
    public String name = null;
    public String addr = null;

    // getter
    public int getStudentId() {
        return studentId;
    }

    public String getStudentName() {
        return name;
    }

    public String getStudentAddr() {
        return addr;
    }

    public void showStudentInfo()
    {
        System.out.println("학번 : " + studentId);
        System.out.println("학생명 : " + name);
        System.out.println("주소 : " + addr);
    }

}
