package datatype;

public class a {
    public static void main(String[] args) {
        int a =10;
        int b =2;
//        b = a - b  + b;
//        a = b + a - a;
        b = b + a - (a=b);
        System.out.println(a);
        System.out.println(b);

        System.out.println("---------");

        a =10;
        b =2;

        b = a - b;
        a = a - b;
        b = b + a;

        System.out.println(a);
        System.out.println(b);
        System.out.println(1 ^ 2);
        System.out.println(2 ^ 2);
        System.out.println(2 ^ 1);

    }
}
