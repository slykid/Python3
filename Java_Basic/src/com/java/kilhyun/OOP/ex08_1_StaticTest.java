package com.java.kilhyun.OOP;

public class ex08_1_StaticTest {

    /*
     * static 변수
     * - 여러 인스턴스가 하나의 값을 공유할 필요가 있다.
     * - 처음 프로그램이 로드 될 때 데이터 영역에 생성되며, 인스턴스의 생성과 상관 없이 사용되기 때무에 클래스 이름으로 참조함
     * - 클래스 변수 혹은 정적 변수라고 부름
     * - 외부에서 값의 변경이 없어야 되기 때문에 일반적으로 private 접근제어자를 붙여서 사용한다.
     *
     * static 메소드
     * - static 변수에 접근하기 위해서 사용하는 메소드를 의미한다.
     * - 클래스 이름으로 참조하여 사용하는 메소드이다.
     * - 클래스 메소드, 정적 메소드 라고도 함
     * - static 메소드에서는 인스턴스 변수를 사용할 수 없다.
     *
     * [변수 별 특징]
     * 변수 유형 		선언위치							사용범위														메모리		생성 및 소멸 시기
     * 지역변수		함수 내부에서 선언						함수 내부에서만 사용												스택			함수가 호출될 때 생성, 함수 종료시 소멸
     * 멤버변수		클래스 멤버변수로 선언					클래스내부에서 사용되고 private이 아는 경우 참조변수로 타 클래스에서 사용가능			힙			인스턴스 생성 시 힙에 생성, 가비지컬렉터가 수거 시 소멸
     * static변수		static 예약어 사용해 클래스 내부에 선언		클래스 내부에서 사용하고 private이 아니면 클래스 이름으로 다른 클래스에서 사용 가능		데이터 영역		프로그램이 처음 시작시 생성, 프로그램 종료시 소멸
     *
     */

    public static void main(String[] args)
    {
        System.out.println("studentKim 추가 시");
        ex08_1_Student_Static studentKim = new ex08_1_Student_Static("Kim");
//		System.out.println(studentKim.sequenceKey + "\n");  // 경고 메세지가 나오는 이유 : static 변수의 경우 객체(인스턴스)의 생성과 무관하기 때문에 정확히 사용하려면 아래 주석의 내용처럼 사용해야됨
//		System.out.println(Ex_Student.sequenceKey + "\n");
        System.out.println(studentKim.getSequenceKey() + "\n");

        System.out.println("studentLee 추가 시");
        ex08_1_Student_Static studentLee = new ex08_1_Student_Static("Lee");
//		System.out.println(studentLee.sequenceKey);
//		System.out.println(studentKim.sequenceKey + "\n"); // studentLee 와 동일한 값이 나오는 이유는 sequenceKey 라는 변수가 같은 메모리를 바라보고 있다는 의미!
        // 스택 메모리는 다르지만, sequenceKey 변수는 실제로 데이터 영역 중 heap 영역에 저장되어 있으며
        // studentKim 과 studentLee 객체가 같은 heap 영역의 sequenceKey를 보고 있다는 의미이다.
        System.out.println(studentLee.getSequenceKey());
        System.out.println(studentKim.getSequenceKey() + "\n");

        System.out.println("StudentID 출력");
        System.out.println(studentKim.getStudentId());
        System.out.println(studentLee.getStudentId());


        // 위의 코드 부분을 모두 주석 처리한 후 아래의 코드를 실행해도 문제가 없음
        // - static 변수의 특 : 객체의 생성과 상관없이 사용 가능함
//		System.out.println(Ex08_1_Student.sequenceKey);
        System.out.println(ex08_1_Student_Static.getSequenceKey());

    }
}
