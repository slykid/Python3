package com.java.kilhyun.OOP;

import java.util.Scanner;

public class ex01_StudentTest {

    public static void main(String[] args)
    {
        // 클래스 생성
        ex01_Student student = new ex01_Student();
        Scanner sc = new Scanner(System.in);

        // 변수에 대입
        System.out.print("학번 : ");
        student.studentId = Integer.parseInt(sc.nextLine().toString());

        System.out.print("이름 : ");
        student.name = sc.nextLine();

        System.out.print("주소 : ");
        student.addr = sc.nextLine();

        System.out.println();

        // 입력 정보 출력
        student.showStudentInfo();
    }
}
