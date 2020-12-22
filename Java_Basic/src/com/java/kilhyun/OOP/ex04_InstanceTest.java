package com.java.kilhyun.OOP;

import java.util.Scanner;

public class ex04_InstanceTest {

    public static void main(String[] args)
    {
        // 클래스 생성
        ex01_Student studentLee = new ex01_Student();
        Scanner sc = new Scanner(System.in);

        studentLee.name = "이순신";
        studentLee.addr = "서울";

        studentLee.showStudentInfo();

        ex01_Student studentKim = new ex01_Student();
        studentKim.name = "김유신";
        studentKim.addr = "경주";

        studentKim.showStudentInfo();

        // 참조 변수이기 때문에 저장되는 값은 16진수의 주소값을 갖는다.
        // 저장은 Heap 메모리에 저장됨
        System.out.println(studentLee);
        System.out.println(studentKim);

    }
}
