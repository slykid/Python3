package com.java.kilhyun.OOP;

public class ex08_1_Student_Static {

    //public static int sequenceKey = 1000; // public 으로 생성 시 문제점 : 외부로부터 변경이 되면 안되는 변수임 -> 반드시 private 으로 선언 해줘야한다!
    private static int sequenceKey = 1000;
    private int studentId;
    public String name = null;
    public String addr = null;

    public ex08_1_Student_Static(String name)
    {
        this.name = name;
        sequenceKey++;
        studentId = sequenceKey;

    }

    public ex08_1_Student_Static(int id, String name)
    {
        this.name = name;
        addr = "주소없음";
        sequenceKey++;
        studentId = sequenceKey;
    }

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

    // static 메소드 생성
    public static int getSequenceKey() {
        return sequenceKey;
    }

    public static void setSequenceKey(int sequenceKey) {
        ex08_1_Student_Static.sequenceKey = sequenceKey;
    }

    // 위의 메소드 중 getSequenceKey() 를 주석처리 후 아래의 코드를 주석 해제한 뒤 확인해보자
//	public static int getSequenceKey() {
//		int i = 0;
//
//		// static 메소드 내에서 인스턴스 변수를 사용하는 것은 불가능
//		// 이유 : static 변수는 객체의 생성과 상관 없이 실행되는 변수이며 프로그램이 로드 되는 시점에 먼저 생성이 되고 프로그램이 종료될 때 사라진다.
//		//      아래의 studentName 은 반드시 Student 라는 객체가 생성되어야 사용이 가능하기 때문이며, static 변수가 생성되는 시점보다 이 후에 생성되기 때문
//		//      일반 메소드에서는 인스턴스 변수를 사용하는 것이 가능함.
//		//      만약 student라는 객체를 static 변수를 생성하는 시점에 생성하게 되면, 가능은 하겠지만 메모리를 많이 차지하기 때문에 자제하는 것이 좋다.
//		//		특히 크기가 큰 Array 등을 static 변수에서 선언하는 경우가 경우가 이에 해당.
//		studentName = "Lee";
//
//		return sequenceKey;
//	}


    // 정보 출력
    public void showStudentInfo()
    {
        System.out.println("학번 : " + studentId);
        System.out.println("학생명 : " + name);
        System.out.println("주소 : " + addr);
    }

}
